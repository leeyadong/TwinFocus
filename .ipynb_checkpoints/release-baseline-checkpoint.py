
import kornia as K

from utils import *

# -----------------------------
# Small IO helpers
# -----------------------------
FIG_DIR = "../result/img/"
NPY_DIR = "../result/npy/"

def _ensure_dirs() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(NPY_DIR, exist_ok=True)

def _save_result(prefix: str, savename: str, tag: str, img: torch.Tensor | np.ndarray) -> None:

    key = f"{prefix}{savename[6:8]}{tag}"

    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = img
    np.save(f"{NPY_DIR}/{key}.npy", np.squeeze(arr))

    # save figure
    plt.figure(figsize=(8 * 0.65, 8))
    plt.pcolor(np.squeeze(arr), cmap="jet")
    plt.axis("off")
    plt.savefig(f"{FIG_DIR}/{key}.png")
    plt.close()


def gradient_descent_ssim(
    scanSet_ref: np.ndarray,
    scanSet_tat: np.ndarray,
    gt_ref: torch.Tensor,
    gt_target: torch.Tensor,
    lr: float = 1e-2,
    epochs: int = 300,
    device: str = "cuda",
    savename: str = "",
    prefix: str = ""
) -> None:

    _ensure_dirs()

    P = scanSet_ref.shape[-1]
    # scanSet_ref = torch.from_numpy(scanSet_ref).to(device)
    # scanSet_tat = torch.from_numpy(scanSet_tat).to(device)
    phi = torch.zeros(P, device=device, requires_grad=True)

    gt_ref = gt_ref.cuda()
    opt = torch.optim.Adam([phi], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    with torch.no_grad():
        tgt_init = focus(scanSet_tat, phi)
        np.save(f"{NPY_DIR}/{prefix}{savename[6:8]}_epoch_0.npy", np.squeeze(tgt_init.detach().cpu().numpy()))

    for _ep in range(epochs):
        opt.zero_grad()

        ref_focused_mag = focus(scanSet_ref, phi)
        ssim_ref = K.metrics.ssim(ref_focused_mag, gt_ref, window_size=11).mean()
        loss = 1 - ssim_ref

        loss.backward()
        opt.step()
        scheduler.step()

    tgt_final = focus(scanSet_tat, phi).cpu().detach().numpy()
    plt.figure(figsize=(8 * 0.65, 8))
    plt.pcolor(np.squeeze(tgt_final), cmap="jet")
    plt.savefig(f"{FIG_DIR}/{prefix}{savename[6:8]}final.png")
    plt.close()
    np.save(f"{NPY_DIR}/{prefix}{savename[6:8]}final.npy", np.squeeze(tgt_final))



def main(filename: str, prefix: str, rand_seed: int) -> None:
    _ensure_dirs()

    with open("./config/radar_config.json") as f:
        config = json.load(f)

    radar_pos_real = np.load("./config/radar_pos_real.npy")
    radar_pos_sim = np.load("./config/radar_pos_sim.npy")

    ref_range = 0.6
    tat_range = 0.6

    letter = filename[6]



    radar_frames_ref = np.load(f"../data/adc_data/ref_real/shape_{letter}.npy")
    radar_frames_tat = np.load(f"../data/adc_data/target_real/{filename}")
    radar_frames_ref_sim = np.load(f"../data/adc_data/ref_sim/sim_adc_{letter}.npy")

    mono_data_ref = convert_multi2mono(radar_frames_ref, config, ref_range)
    mono_data_tat = convert_multi2mono(radar_frames_tat, config, tat_range)

    points = np.load(f"../data/point_cloud/{filename}")
    center = points.mean(axis=0)

    # Active overrides (exactly as your current code)
    center_offset = center[:-1] - [0.0174, 0.0677]
    y = np.array([-0.05, 0.09]) + center_offset[1]
    y = np.array([-0.07, 0.11])
    y_sim = np.array([-0.05, 0.09])
    y_sim = np.array([-0.07, 0.11])
    x_tat = np.array([0.03, 0.12])

    x_ref = np.array([-0.09, -0.01]) - center_offset[0]
    x_ref = np.array([-0.11, 0.01])
    x_sim = np.array([-0.09, -0.01])
    x_sim = np.array([-0.11, 0.01])

    _offset = np.array([0.05, -0.015])


    #
    sim_gt, _ = sar_bp_conv_block(radar_frames_ref_sim, [radar_pos_sim, radar_pos_sim], config, ref_range,
                                  [x_sim+_offset[0], y_sim+_offset[1]], (100, 100), 20)
    
    ref_gt, _ = sar_bp_conv_block(mono_data_ref, [radar_pos_real, radar_pos_real], config, ref_range,
                                  [x_ref, y], (100, 100), 20)

    tat_gt, _ = sar_bp_conv_block(mono_data_tat, [radar_pos_real, radar_pos_real], config, tat_range,
                                  [x_tat, y], (100, 100), 20)


    sim_gt = soft_thr(sim_gt)
    tat_gt = soft_thr(tat_gt)
    ref_gt = soft_thr(ref_gt)

    ref_defocus, scanset_ref = sar_bp_conv_block(mono_data_ref, [radar_pos_real, radar_pos_real], config, ref_range,
                                  [x_ref, y], (100, 100), 20, sim_err=True, frame_block=86, randseed=rand_seed,exp="shape")
    tat_defocus, scanset_tat = sar_bp_conv_block(mono_data_tat, [radar_pos_real, radar_pos_real], config, tat_range,
                                  [x_tat, y], (100, 100), 20,  sim_err=True, frame_block=86,randseed=rand_seed,exp="shape")
    #tat_defocus = soft_thr(tat_defocus)
    #sample = np.stack([tat_defocus.cpu().numpy(), tat_gt.cpu().numpy()], axis=-1)
    #np.save("../data/ifnet_train/"+filename[6]+"_"+str(rand_seed)+".npy", sample)
    gradient_descent_ssim(scanset_ref,
                          scanset_tat,
                          sim_gt,
                          tat_gt,
                          lr=5e-2,
                          epochs=300, savename=filename[:-4], prefix="sim" + prefix)
    
    gradient_descent_ssim(scanset_ref,
                          scanset_tat,
                          ref_gt,
                          tat_gt,
                          lr=5e-2,
                          epochs=300, savename=filename[:-4], prefix="real" + prefix)
    
    torch.cuda.empty_cache()
    plt.close()


def main_loop(rand_seed: int) -> None:
    fnames = os.listdir("../data/adc_data/target_real/")

    for f in fnames:
        print(f)
        if f.find("shape")==-1:
            continue
        main(f, f"newseed{rand_seed}shape_", rand_seed)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    for i in range(2020, 2030):
        print(i)
        main_loop(i)