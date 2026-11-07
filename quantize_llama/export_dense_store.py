"""Write no-finetune QTIP weights as a dense per-tensor store.

Each tensor is quantized independently from the original weight and a stored
Hessian, so tensors from different bit-widths can be recombined freely. The
trellis is still packed, because its size is the tensor's physical cost.

The output follows the prism-placement-store-v1 layout read by PRISM.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lib import utils
from lib.algo import ldlq
from lib.codebook import bitshift

STORE_SCHEMA = 'prism-placement-store-v1'
TLUT_CACHE = '/tmp/kmeans_9_2.pt'
L, V, TLUT_BITS, DECODE_MODE, TD = 16, 2, 9, 'quantlut_sym', 16
# module, Hessian name: the same order as quantize_finetune_llama.
MODULES = (('self_attn.v_proj', 'qkv'), ('self_attn.q_proj', 'qkv'),
           ('self_attn.k_proj', 'qkv'), ('self_attn.o_proj', 'o'),
           ('mlp.up_proj', 'up'), ('mlp.gate_proj', 'up'),
           ('mlp.down_proj', 'down'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir')
    parser.add_argument('--hess-dir')
    parser.add_argument('--out', required=True)
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--layers', type=int, nargs='*', default=[])
    parser.add_argument('--modules', nargs='*', default=None)
    parser.add_argument('--tlut-file')
    parser.add_argument('--sigma-reg', type=float, default=1e-2)
    parser.add_argument('--scale-override', type=float, default=0.9)
    parser.add_argument('--check-decode', action='store_true')
    parser.add_argument('--prepare-tlut', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--metadata-json', default='{}')
    return parser.parse_args()


def codebook(bits):
    return bitshift.bitshift_codebook(L=L, K=bits, V=V, tlut_bits=TLUT_BITS,
                                      decode_mode=DECODE_MODE)


def prepare_tlut(path):
    torch.manual_seed(0)
    np.random.seed(0)
    if os.path.exists(TLUT_CACHE):
        os.remove(TLUT_CACHE)
    codebook(2)
    shutil.copyfile(TLUT_CACHE, path)


class Weights:

    def __init__(self, model_dir):
        self.root = Path(model_dir)
        index = json.loads((self.root / 'model.safetensors.index.json').read_text())
        self.shards = index['weight_map']

    def get(self, key):
        with safe_open(self.root / self.shards[key], framework='pt') as shard:
            return shard.get_tensor(key)


def load_hessian(hess_dir, idx, name):
    data = torch.load(f'{hess_dir}/{idx}_{name}.pt', map_location='cpu')
    hessian = utils.flat_to_sym(data['flatH'], data['n'])
    if 'mu' in data:
        hessian += data['mu'][None, :] * data['mu'][:, None]
    return hessian.to(torch.float32)


def forward_transform(W, SU, SV):
    return utils.matmul_hadUt(utils.matmul_hadUt(W.T * SV).T * SU)


def inverse_transform(Wr, SU, SV):
    return (utils.matmul_hadU((utils.matmul_hadU(Wr) * SU).T) * SV).T


def pack(Qidxs, cb, m, n, bits):
    """Pack indices exactly as quantize_finetune_decoder_layer does for the kernel."""
    packed = cb.pack_trellis(
        Qidxs.cpu().reshape(m // TD, TD, n // TD, TD // V).transpose(1, 2).reshape(
            -1, TD * TD // V))
    return packed.view(torch.uint8).view(-1, 2).flip((-1, )).reshape(
        m // 16 // 2, 2, n // 16 // 2, 2, 16 * 16 // 8,
        bits).permute(0, 2, 4, 3, 1, 5).flip((-1, )).contiguous().flatten().view(
            torch.int16).reshape(packed.shape)


def decode_gap(packed, hatWr, cb, m, n, bits, device, dtype):
    """Largest gap between decoding the packed bits and the captured weights."""
    linear = bitshift.BitshiftLinear(TD, TD, L, bits, V, TLUT_BITS, DECODE_MODE,
                                     dtype=dtype, tlut=cb.tlut, has_kernel=True)
    decoded = linear.get_hatW_kernel(packed.to(device), m, n)
    return (decoded.float() - hatWr.float()).abs().max().item()


def quantize_tensor(W_orig, hessian, cb, bits, seed_value, args, device):
    torch.manual_seed(seed_value)
    W = W_orig.to(device=device, dtype=torch.float32)
    m, n = W.shape
    SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(torch.float32)
    SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(torch.float32)

    HR = utils.regularize_H(hessian.clone(), args.sigma_reg).to(device)
    HRr = utils.matmul_hadUt(utils.matmul_hadUt(HR * SU).T * SU)
    Wr = forward_transform(W, SU, SV)
    roundtrip = (inverse_transform(Wr, SU, SV) - W).norm() / W.norm()
    if roundtrip.item() > 1e-4:
        raise RuntimeError(f'Incoherence transform does not invert: {roundtrip.item()}')

    Wscale = Wr.square().mean().sqrt() / (
        cb.lut.to(torch.float64).square().mean().sqrt().float() * args.scale_override)
    Wr = Wr / Wscale
    LRr, _ = utils.block_LDL(HRr, TD)
    diag = torch.arange(n, device=LRr.device)
    LRr[diag, diag] = 0
    ldlq_args = SimpleNamespace(td_x=TD, td_y=TD, V=V)
    hatWr, Qidxs = ldlq.LDLQ(Wr, LRr, cb, ldlq_args, for_kernel=True)
    packed = pack(Qidxs, cb, m, n, bits)

    gap = None
    if args.check_decode:
        gap = decode_gap(packed, hatWr, cb, m, n, bits, device, W_orig.dtype)

    Wr, hatWr = Wr * Wscale, hatWr * Wscale
    proxy_err = (torch.trace((Wr - hatWr) @ HRr @ (Wr - hatWr).T) /
                 torch.trace(Wr @ HRr @ Wr.T)).item()
    decoded = inverse_transform(hatWr, SU, SV).to(W_orig.dtype).cpu()
    reference = W_orig.float()
    relative_error = ((reference - decoded.float()).norm() / reference.norm()).item()
    # Trellis, the two sign vectors in the model dtype, and this layer's copy of the LUT.
    stored_bytes = packed.numel() * 2 + (m + n) * 2 + cb.tlut.numel() * 2
    return decoded, stored_bytes, relative_error, proxy_err, gap


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    device = torch.device('cuda')
    if args.tlut_file:
        shutil.copyfile(args.tlut_file, TLUT_CACHE)
    weights = Weights(args.model_dir)
    out = Path(args.out)
    modules = [item for item in MODULES
               if args.modules is None or item[0].split('.')[-1] in args.modules]
    for idx in args.layers:
        hessians = {}
        for position, (module, hess_name) in enumerate(modules):
            layer_name = f'model.layers.{idx}.{module}'
            for bits in args.bits:
                row_path = out / f'rows/seed{args.seed}/{bits}bit/{layer_name}.json'
                if row_path.exists():
                    continue
                if hess_name not in hessians:
                    hessians = {hess_name: load_hessian(args.hess_dir, idx, hess_name)}
                cb = codebook(bits).to(device)
                W = weights.get(f'{layer_name}.weight')
                cb = cb.to(W.dtype)
                started = time.time()
                seed_value = args.seed * 1000003 + idx * 97 + position
                decoded, stored_bytes, relative_error, proxy_err, gap = quantize_tensor(
                    W, hessians[hess_name], cb, bits, seed_value, args, device)
                relative = f'tensors/seed{args.seed}/{bits}bit/{layer_name}.safetensors'
                path = out / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file({'weight': decoded.contiguous()}, path)
                row = {
                    'layer_name': layer_name,
                    'bits': bits,
                    'seed': args.seed,
                    'path': relative,
                    'sha256': file_sha256(path),
                    'shape': list(decoded.shape),
                    'stored_bytes': int(stored_bytes),
                    'relative_error': relative_error,
                    'proxy_err': proxy_err,
                    'decode_gap': gap,
                    'seconds': time.time() - started,
                }
                row_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = row_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(row))
                temporary.replace(row_path)
                print(json.dumps({key: row[key] for key in (
                    'layer_name', 'bits', 'stored_bytes', 'relative_error', 'proxy_err',
                    'decode_gap', 'seconds')}), flush=True)
                del cb, decoded
                utils.clean()


def finalize(args):
    out = Path(args.out)
    rows = [json.loads(path.read_text()) for path in sorted(out.glob('rows/**/*.json'))]
    manifest = {
        'schema_version': STORE_SCHEMA,
        'metadata': json.loads(args.metadata_json),
        'tensors': rows,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f'Wrote manifest with {len(rows)} tensors')


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    # Upstream compiles one K per process. Here K changes within a process, and
    # inductor cannot build the dynamic 2**(K*V) graph it would otherwise choose.
    torch._dynamo.config.automatic_dynamic_shapes = False
    if args.prepare_tlut:
        prepare_tlut(args.out)
    elif args.finalize:
        finalize(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
