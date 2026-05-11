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
        density_dir=Path('Densitymaps'),
        filterbank_dir=Path('filterbank_packed.npz'),
        noise_blob_dir=Path('noise_blob'),
        strict_assets=True,
        filter_zero_point=46,
    )
    
    for i in range(7):
        print(f"Generating master {i+1}/100...")
        
        # Reset generator dimensions for master generation
        gen.W = RENDER_W
        gen.H = RENDER_H
        
        current_out_dir = out_dir / f'fingerprint_{i:03d}'
        
        master = gen.generate_master(class_distribution=i, save_debug=str(current_out_dir / 'master_debug'), out_size=None)
        
        master_small = cv2.resize(master, OUT_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(current_out_dir / 'master_debug' / 'master_only_256.png'), master_small)
            
        gen.save_metadata(current_out_dir / 'master_debug' / 'metadata.txt')

        # Set smaller dimensions for impression distortion
        gen.W = TARGET_W
        gen.H = TARGET_H

        impressions = gen.generate_impressions(
            master_img=master_small, 
            out_dir=current_out_dir / 'impressions',
            n_impr=3,
            min_noise_level=0,
            max_noise_level=0,
            save_debug=True,
            out_size=None, 
        )
        
        print(f'Saved {len(impressions)} impressions to: {current_out_dir / "impressions"}')