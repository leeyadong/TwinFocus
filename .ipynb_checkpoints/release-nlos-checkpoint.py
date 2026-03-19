
import kornia as K
from utils import *

FIG_DIR = "../results/img/"

def _ensure_dirs() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

def gradient_descent_ssim(
    scanSet_ref: torch.Tensor,
    scanSet_tat: torch.Tensor,
    gt_ref: torch.Tensor,
    gt_tgt: torch.Tensor,
    lr: float = 1e-2,
    epochs: int = 300,
    device: str = "cuda",
    savename: str = "",
) -> None:

    _ensure_dirs()

    P = scanSet_ref.shape[-1]
    phi = torch.zeros(P, device=device, requires_grad=True)

    gt_ref = gt_ref.cuda()
    opt = torch.optim.Adam([phi], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    with torch.no_grad():
        tgt_init = focus(scanSet_tat, phi)
        ssim_tgt_init = K.metrics.ssim(tgt_init, gt_tgt, window_size=11).mean()
        ref_init = focus(scanSet_ref, phi)
        ssim_ref_init = K.metrics.ssim(ref_init, gt_ref, window_size=11).mean()
    
    for _ep in range(epochs):
        opt.zero_grad()

        ref_focused_mag = focus(scanSet_ref, phi)
        ssim_ref = K.metrics.ssim(ref_focused_mag, gt_ref, window_size=11).mean()
        loss = 1 - ssim_ref

        loss.backward()
        opt.step()
        scheduler.step()

    tgt_final = focus(scanSet_tat, phi)
    ssim_tgt_final = K.metrics.ssim(tgt_final, gt_tgt, window_size=11).mean()

    plt.figure(figsize=(12, 8))

    if savename.find("physical_twin")!=-1:
        ref_type = "Physical Twin"

    if savename.find("digital_twin")!=-1:
        ref_type = "Digital Twin"
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # ---------- Row 1: Reference ----------
    axes[0, 0].pcolor(np.squeeze(gt_ref.detach().cpu().numpy()), cmap="jet")
    axes[0, 0].set_title(f"Ground-Truth Reference\n({ref_type})", fontsize=11)
    
    axes[0, 1].pcolor(np.squeeze(ref_init.detach().cpu().numpy()), cmap="jet")
    axes[0, 1].set_title(
        f"Defocused Reference with Motion Error\nSSIM = {ssim_ref_init.detach().cpu().item():.2f}",
        fontsize=11
    )
    
    axes[0, 2].pcolor(np.squeeze(ref_focused_mag.detach().cpu().numpy()), cmap="jet")
    axes[0, 2].set_title(
        f"Refocused Reference with TwinFocus\nSSIM = {ssim_ref.detach().cpu().item():.2f}",
        fontsize=11
    )
    
    # ---------- Row 2: Target ----------
    axes[1, 0].pcolor(np.squeeze(gt_tgt.detach().cpu().numpy()), cmap="jet")
    axes[1, 0].set_title("Ground-Truth Target", fontsize=11)
    
    axes[1, 1].pcolor(np.squeeze(tgt_init.detach().cpu().numpy()), cmap="jet")
    axes[1, 1].set_title(
        f"Defocused Target with Motion Error\nSSIM = {ssim_tgt_init.detach().cpu().item():.2f}",
        fontsize=11
    )
    
    axes[1, 2].pcolor(np.squeeze(tgt_final.detach().cpu().numpy()), cmap="jet")
    axes[1, 2].set_title(
        f"Refocused Target with TwinFocus\nSSIM = {ssim_tgt_final.detach().cpu().item():.2f}",
        fontsize=11
    )
    
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    
 
    fig.suptitle(f"TwinFocus Refocusing Results: Reference: {savename[5]}, Target: {savename[6]}, Mode: {ref_type}, Occlusion: {savename.split('_')[2]}", fontsize=14)
    
    plt.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.88,
        bottom=0.06,
        wspace=0.18,
        hspace=0.35
    )
    
    plt.savefig(f"{FIG_DIR}/{savename}.png", dpi=300, bbox_inches="tight")
    plt.close()



def main(filename: str, rand_seed=2026) -> None:
    _ensure_dirs()
    
    # load radar and imaging configurations
    with open("./config/radar_config.json") as f:
        config = json.load(f)
    imaging_cfg = config["imaging"]
    radar_pos_real = np.load("./config/radar_pos_real.npy")
    radar_pos_sim = np.load("./config/radar_pos_sim.npy")
    ref_range = 0.6
    tat_range = 0.76

    letter = filename[5] # the reference letter
    print("Processing File: ", filename)

    # load radar adc data
    if filename.find("foam") != -1:
        radar_frames_ref = np.load(f"../data/adc_data/ref_real/nlos_{letter}_foam.npy")
    else:
        radar_frames_ref = np.load(f"../data/adc_data/ref_real/nlos_{letter}_other.npy")

    radar_frames_tat = np.load(f"../data/adc_data/target_real/{filename}")
    radar_frames_ref_sim = np.load(f"../data/adc_data/ref_sim/sim_adc_{letter}.npy")
    
    # convert multistatic radar data to monostatic data
    mono_data_ref = convert_multi2mono(radar_frames_ref, config, ref_range)
    mono_data_tat = convert_multi2mono(radar_frames_tat, config, tat_range)

    # load point clouds of the reference captured by the camera for spatial alignment
    points = np.load(f"../data/point_cloud_nlos/{filename}")
    center = points.mean(axis=0)

    def arr(key):
        return np.array(imaging_cfg[key], dtype=float)

    # imaging area for the reference and target
    y = arr("target_y_bounds") + center[1]
    y_sim = arr("sim_y_bounds")
    x_tat = arr("target_x_bounds")
    x_ref = arr("reference_x_bounds") - center[0]
    x_sim = arr("sim_x_bounds")
    sim_offset = arr("sim_offset")

    # bp imaging with GT radar positions to get the well-focused images
    sim_gt, _ = sar_bp_conv_block(radar_frames_ref_sim, [radar_pos_sim, radar_pos_sim], config, ref_range,
                                  [x_sim+sim_offset[0], y_sim+sim_offset[1]], (100, 100), 20)

    ref_gt, _ = sar_bp_conv_block(mono_data_ref, [radar_pos_real, radar_pos_real], config, ref_range,
                                  [x_ref, y], (100, 100), 20)

    tat_gt, _ = sar_bp_conv_block(mono_data_tat, [radar_pos_real, radar_pos_real], config, tat_range,
                                  [x_tat, y], (100, 100), 20)

    sim_gt = soft_thr(sim_gt)
    tat_gt = soft_thr(tat_gt)
    ref_gt = soft_thr(ref_gt)

    # bp imaging with tracking errors to get the defocused image
    ref_defocus, scanset_ref = sar_bp_conv_block(mono_data_ref, [radar_pos_real, radar_pos_real], config, ref_range,
                                  [x_ref, y], (100, 100), 20, sim_err=True, frame_block=86, randseed=rand_seed)
    tat_defocus, scanset_tat = sar_bp_conv_block(mono_data_tat, [radar_pos_real, radar_pos_real], config, tat_range,
                                  [x_tat, y], (100, 100), 20,  sim_err=True, frame_block=86,randseed=rand_seed)

    # perform ssim optimization using gradient descent
    # first in digital_twin mode: use simulated reference template
    gradient_descent_ssim(scanset_ref,
                          scanset_tat,
                          sim_gt,
                          tat_gt,
                          lr=5e-2,
                          epochs=300, savename=filename[:-4]+"_digital_twin")
    
    # then in physical_twin mode: use measured reference template
    gradient_descent_ssim(scanset_ref,
                          scanset_tat,
                          ref_gt,
                          tat_gt,
                          lr=5e-2,
                          epochs=300, savename=filename[:-4]+"_physical_twin")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    fnames = os.listdir("../data/adc_data/target_real/")
    for f in fnames:
        if f.find("nlos") == -1:
            continue
        main(f)
