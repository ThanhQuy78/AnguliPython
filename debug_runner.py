from pathlib import Path

import cv2

from pipeline import AnguliFaithfulGenerator


if __name__ == '__main__':
    base = Path(r'')
    out_dir = base / 'pipeline_debug_faithful_v3'

    # ── Super-resolution rendering ────────────────────────────────────────────
    # Generate at 2× canvas size (512×720) so the fixed-size filterbank kernels
    # produce ridges that are 2× wider in pixel space.  Downscaling with
    # INTER_AREA then halves the ridge width → twice as many ridges per cm.
    # Seed count scales automatically with canvas area (×4 here).
    RENDER_SCALE = 2
    TARGET_W, TARGET_H = 256, 360
    RENDER_W, RENDER_H = TARGET_W * RENDER_SCALE, TARGET_H * RENDER_SCALE
    OUT_SIZE = (TARGET_W, TARGET_H)   # (W, H) for cv2.resize
    # ─────────────────────────────────────────────────────────────────────────

    from pathlib import Path

    gen = AnguliFaithfulGenerator(
        W=RENDER_W,
        H=RENDER_H,
        generation_seed=42,
        density_dir=Path('/kaggle/input/datasets/poseidon127/anguli-packed/Densitymaps'),
        filterbank_dir=Path('/kaggle/input/datasets/poseidon127/anguli-packed/filterbank_packed.npz'),
        noise_blob_dir=Path('/kaggle/input/datasets/poseidon127/anguli-packed/noise_blob'),
        strict_assets=True,
        filter_zero_point=46,
    )
    """
    info = gen.run_until_global_filter_and_save(
            class_distribution=6,
            out_dir=str(out_dir / 'pipeline_steps'),
        )
    """
    master = gen.generate_master(class_distribution=6, save_debug=str(out_dir / 'master_debug'), out_size=None)
    
    master_small = cv2.resize(master, OUT_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / 'master_debug' / 'master_only_256.png'), master_small)
        
    gen.save_metadata(out_dir / 'master_debug' / 'metadata.txt')

    gen.W = TARGET_W
    gen.H = TARGET_H

    impressions = gen.generate_impressions(
        master_img=master_small, 
        out_dir=out_dir / 'impressions',
        n_impr=4,
        min_noise_level=0,
        max_noise_level=0,
        save_debug=True,
        out_size=None, 
    )
    # ----------------------------
    
    print(f'Saved {len(impressions)} impressions to: {out_dir / "impressions"}')