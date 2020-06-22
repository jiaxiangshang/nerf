#
#
from baselib_python.Geometry.Camera.np_rotation import get_eye, ext_to_rot_t, get_opengl_camAxis
from baselib_python.IO.BlendMVS import load_cam, parse_pair_txt
from tf_viewSyn.nerf.load_data.load_llff import *


def load_blmvs_data(scene='cube', basedir='/data/deepvoxels', interv=10, mask=False):
    dic_base = '{}/{}/'.format(basedir, scene)

    path_pair = os.path.join(dic_base, 'cams', 'pair.txt')
    path_cam_pattern = os.path.join(dic_base, 'cams', '%.8d_cam.txt')
    if mask:
        pass
    else:
        path_img_pattern = os.path.join(dic_base, 'blended_images', '%.8d.jpg')

    num_render = parse_pair_txt(path_pair)
    list_c2w = []
    list_cams = []
    list_imgs = []

    list_val_c2w = []
    list_val_cams = []
    list_val_imgs = []
    for i in num_render:
        cam = load_cam(path_cam_pattern%i)
        img = imageio.imread(path_img_pattern%i) / 255.
        #
        H, W, _ = img.shape
        focal = cam[1][0][0]
        hwf = [H, W, focal]
        hwf_np = np.array(hwf)
        hwf_np = np.expand_dims(hwf_np, -1)
        if 0:
            # w2c convert to c2w
            rot, trans = ext_to_rot_t(cam[0])
            c2w_opengl = get_opengl_camAxis(rot, trans)

            rot_c2w, trans_c2w = ext_to_rot_t(c2w_opengl)
            position = get_eye(rot_c2w, trans_c2w)
            position = np.expand_dims(position, -1)
            c2w = np.concatenate([np.transpose(rot_c2w), position, hwf_np], axis=1)
        else:
            # rot, trans = ext_to_rot_t(cam[0])
            # position = get_eye(rot, trans)
            # position = np.expand_dims(position, -1)
            # c2w = np.concatenate([np.transpose(rot), position, hwf_np], axis=1)
            c2w = np.linalg.inv(cam[0])[:3, :4]
            c2w = np.concatenate([c2w[:, 0:1], -c2w[:, 1:2], -c2w[:, 2:]], 1)
            c2w = np.concatenate([c2w, hwf_np], axis=1)

        #c2w = np.concatenate([rot, position, hwf_np], axis=1)
        #cam[2][:3, :] = c2w[:3, :4]
        #
        if i % interv == 0:
            list_val_imgs.append(img)
            list_val_cams.append(cam)
            list_val_c2w.append(c2w)
        else:
            list_imgs.append(img)
            list_cams.append(cam)
            list_c2w.append(c2w)

    #
    all_imgs = [list_imgs, list_val_imgs, list_val_imgs]
    all_cams = list_cams + list_val_cams + list_val_cams
    all_c2ws = list_c2w + list_val_c2w + list_val_c2w
    counts = [0] + [len(x) for x in all_imgs]
    counts = np.cumsum(counts)
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    all_imgs = list_imgs + list_val_imgs + list_val_imgs
    stk_imgs = np.stack(all_imgs, 0).astype(np.float32)
    stk_cams = np.stack(all_cams, 0).astype(np.float32)
    stk_c2ws = np.stack(all_c2ws, 0).astype(np.float32)
    print('load_blmvs_data: img shape, cam shape', stk_imgs.shape, stk_cams.shape)

    stk_depth_min = stk_cams[:, 1, 3, 0].min()
    stk_depth_max = stk_cams[:, 1, 3, 3].max()
    if 0:
        # generate render pose
        c2w = poses_avg(stk_c2ws)
        print('load_blmvs_data: recentered cam pos', c2w.shape)
        print(c2w[:3, :4])

        ## Get spiral
        # Get average pose
        up = normalize(stk_c2ws[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        stk_depth_min = stk_cams[:, 1, 3, 0].min()
        stk_depth_max = stk_cams[:, 1, 3, 3].max()

        close_depth, inf_depth = stk_depth_min * .9, stk_depth_max * 5.
        dt = .75
        mean_dz = 1. / ((1. - dt) / close_depth + dt / inf_depth)
        focal_render = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = stk_c2ws[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 240
        N_rots = 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal_render, zdelta, zrate=.5, rots=N_rots, N=N_views)
        render_poses = np.array(render_poses).astype(np.float32)
    else:
        bds = np.stack([stk_cams[:, 1, 3, 0], stk_cams[:, 1, 3, 3]], axis=0)

        stk_c2ws = recenter_poses(stk_c2ws)

        stk_c2ws, render_poses, bds = spherify_poses(stk_c2ws, bds)

    return stk_imgs, stk_c2ws, render_poses, hwf, i_split, stk_depth_min, stk_depth_max


