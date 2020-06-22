import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import ast
import time


#self
_curr_path = os.path.abspath(__file__) # /home/..../face
_cur_dir = os.path.dirname(_curr_path) # ./
_tf_dir = os.path.dirname(_cur_dir) # ./
_deep_learning_dir = os.path.dirname(_tf_dir) # ../
print(_deep_learning_dir)
sys.path.append(_deep_learning_dir) # /home/..../pytorch3d

from tf_viewSyn.nerf.run_nerf_helpers import *
from tf_viewSyn.nerf.load_data.load_llff import load_llff_data
from tf_viewSyn.nerf.load_data.load_deepvoxels import load_dv_data
from tf_viewSyn.nerf.load_data.load_blender import load_blender_data
from tf_viewSyn.nerf.load_data.load_blmvs import load_blmvs_data

from tf_viewSyn.nerf.build_graph_ray import *
from tf_viewSyn.nerf.build_graph_patch import *

tf.compat.v1.enable_eager_execution()


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options: net design
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')

    # training options: lr
    parser.add_argument("--max_global_steps", type=int, default=1000000,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    # training options: ray sample
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory, (2 min than nerf)')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory (2 min than nerf)')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')
    
    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')    

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    #
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels / blmvs')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=50000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    # jiaxiang
    parser.add_argument("--flag_patch_ray", type=ast.literal_eval, default=False,
                        help='frequency of render_poses video saving')
    parser.add_argument("--pr_patch_size", type=int, default=12,
                        help='frequency of render_poses video saving')

    return parser

"""
python run_nerf.py --config ./cascade_config/blendmvs_success/config_city_lib.txt \
--no_ndc --spherify --lindisp

# debug
python run_nerf.py --config ./cascade_config/config_debug.txt \
--no_ndc --spherify --lindisp

python run_nerf.py --config ./cascade_config/LOCALconfig_city_temple.txt \
--no_ndc --spherify --lindisp

python run_nerf.py --config ./config/paper_configs/0_reproduce_2048/LOCALconfig_rs_fern.txt --no_ndc --spherify --lindisp

python run_nerf.py --config ./config/paper_configs/0_reproduce_2048/LOCALconfig_rs_fern_patchRay.txt --no_ndc --spherify --lindisp

# 0 reproduce 2048
bash

# 1 patch 2048
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config ./config/paper_configs/1_patch_2048/config0_patch_fern.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config ./config/paper_configs/1_patch_2048/config4_patch_leaves.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config ./config/paper_configs/1_patch_2048/config4_patch12_leaves.txt \
--no_ndc --spherify --lindisp


# 8-card
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config ./cascade_config/blendmvs_success/config_city_lib.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config ./cascade_config/blendmvs_success/config_city_tingzi.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config ./cascade_config/blendmvs_success/config_obj_ball.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config ./cascade_config/blendmvs_success/config_obj_bread.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=6 python run_nerf.py --config ./cascade_config/blendmvs_success/config_obj_camera.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=7 python run_nerf.py --config ./cascade_config/blendmvs_success/config_obj_gate.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config ./cascade_config/blendmvs_success/config_city_yunnan.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config ./cascade_config/blendmvs_success/config_obj_car.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config ./cascade_config/blendmvs_success/config_obj_rock.txt \
--no_ndc --spherify --lindisp


"""
def train():

    parser = config_parser()
    args = parser.parse_args()

    # Random seed
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # Load data
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = tf.reduce_min(bds) * .9
            far = tf.reduce_max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]
    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.
    elif args.dataset_type == 'blmvs':
        images, poses, render_poses, hwf, i_split, stk_depth_min, stk_depth_max = load_blmvs_data(
            scene=args.shape, basedir=args.datadir, interv=args.testskip
        )
        poses = poses[:, :3, :4]
        print('Loaded blmvs', images.shape, render_poses.shape, hwf, args.datadir)
        #
        i_train, i_val, i_test = i_split
        #
        near = stk_depth_min-10.
        far = stk_depth_max+10.
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    path_f_eval = os.path.join(basedir, expname, 'eval.txt')
    f_eval = open(path_f_eval, 'w')

    # Create nerf model
    if args.flag_patch_ray:
        render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf_patch(args)
    else:
        render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(args)

    bds_dict = {
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        if args.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
            'test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                              gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
        print('Done rendering', testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                         to8b(rgbs), fps=30, quality=8)

        return

    # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(
            lrate, decay_steps=args.lrate_decay * 1000, decay_rate=0.1
        )
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    pr_N_rand = int(args.N_rand / (args.pr_patch_size*args.pr_patch_size))

    use_batching = not args.no_batching
    if use_batching:
        if args.flag_patch_ray:
            rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
            rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
            print('Prepare raybatch tensor: 0. generate rays(finish)')
            # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
            # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
            rays_rgb = np.stack([rays_rgb[i] for i in i_train], axis=0)  # train images only
            print('Prepare raybatch tensor: 1. concat image and select training set(finish)')

            list_pr = []
            for i in range(int(H / args.pr_patch_size)):
                for j in range(int(W / args.pr_patch_size)):
                    i_jump = i * args.pr_patch_size
                    j_jump = j * args.pr_patch_size
                    pr = rays_rgb[:, i_jump:i_jump+args.pr_patch_size, j_jump:j_jump+args.pr_patch_size, :, :]
                    list_pr.append(pr)
            patchRays_rgb = np.stack(list_pr, axis=1) # [N, H * W / pr_patch_size^2, pr_patch_size^2, ro+rd+rgb, 3]
            print('Prepare patchRaybatch tensor: 2. generate patch(finish)')


            # flat: [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = np.reshape(patchRays_rgb, [-1, args.pr_patch_size, args.pr_patch_size, 3, 3])
            rays_rgb = rays_rgb.astype(np.float32)

            np.random.shuffle(rays_rgb)
            print('Prepare raybatch tensor: 3. shuffle training set(finish)')
            i_batch = 0
        else:
            # For random ray batching.
            #
            # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
            # interpreted as,
            #   axis=0: ray origin in world space
            #   axis=1: ray direction in world space
            #   axis=2: observed RGB color of pixel
            # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
            # for each pixel in the image. This stack() adds a new dimension.
            rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
            rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
            print('Prepare raybatch tensor: 0. generate rays(finish)')
            # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
            # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
            rays_rgb = np.stack([rays_rgb[i] for i in i_train], axis=0)  # train images only
            print('Prepare raybatch tensor: 0. concat image and select training set(finish)')
            # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
            rays_rgb = rays_rgb.astype(np.float32)
            np.random.shuffle(rays_rgb)
            print('Prepare raybatch tensor: 0. shuffle training set(finish)')
            i_batch = 0

    N_iters = args.max_global_steps
    #N_iters = 5000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = tf.contrib.summary.create_file_writer(os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    for i in range(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            if args.flag_patch_ray:
                # Random over all images
                batch = rays_rgb[i_batch:i_batch+pr_N_rand]  # [B, 2+1, 3*?]
                batch = tf.transpose(batch, [3, 0, 1, 2, 4])

                # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
                # target_s[n, rgb] = example_id, observed color.
                batch_rays, target_s = batch[:2], batch[2]

                i_batch += pr_N_rand
            else:
                # Random over all images
                batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
                batch = tf.transpose(batch, [1, 0, 2])

                # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
                # target_s[n, rgb] = example_id, observed color.
                batch_rays, target_s = batch[:2], batch[2]

                i_batch += N_rand

            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0
        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, pose)
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H//2 - dH, H//2 + dH), 
                        tf.range(W//2 - dW, W//2 + dW), 
                        indexing='ij'), -1)
                    if i < 10:
                        print('precrop', dH, dW, coords[0,0], coords[-1,-1])
                else:
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H), tf.range(W), indexing='ij'), -1)
                coords = tf.reshape(coords, [-1, 2])
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False)
                select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
                rays_o = tf.gather_nd(rays_o, select_inds)
                rays_d = tf.gather_nd(rays_d, select_inds)
                batch_rays = tf.stack([rays_o, rays_d], 0)
                target_s = tf.gather_nd(target, select_inds)

        #####  Core optimization loop  #####
        with tf.GradientTape() as tape:

            # Make predictions for color, disparity, accumulated opacity.
            if args.flag_patch_ray:
                rgb, disp, acc, extras = render_patch(
                    H, W, focal, chunk=int(args.chunk / (args.pr_patch_size*args.pr_patch_size)), rays=batch_rays,
                    verbose=i < 10, retraw=True, pr_patch_size=args.pr_patch_size, **render_kwargs_train)

                # Compute MSE loss between predicted and true RGB.
                img_loss = img2mse(rgb, target_s)
                trans = extras['raw'][..., -1]
                loss = img_loss
                psnr = mse2psnr(img_loss)

                # Add MSE loss for coarse-grained model
                if 'rgb0' in extras:
                    img_loss0 = img2mse(extras['rgb0'], target_s)
                    loss += img_loss0
                    psnr0 = mse2psnr(img_loss0)
            else:
                rgb, disp, acc, extras = render(
                    H, W, focal, chunk=args.chunk, rays=batch_rays,
                    verbose=i < 10, retraw=True, **render_kwargs_train)

                # Compute MSE loss between predicted and true RGB.
                img_loss = img2mse(rgb, target_s)
                trans = extras['raw'][..., -1]
                loss = img_loss
                psnr = mse2psnr(img_loss)

                # Add MSE loss for coarse-grained model
                if 'rgb0' in extras:
                    img_loss0 = img2mse(extras['rgb0'], target_s)
                    loss += img_loss0
                    psnr0 = mse2psnr(img_loss0)

        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        dt = time.time()-time0

        #####           end            #####
        # Rest is logging
        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_print == 0 or i < 10:
            print('Exp({:s}) | Iter({:d}) / Time({:.05f}) / psnr({:.05f}) / loss({:.05f})'.format(expname, global_step.numpy(), dt, psnr.numpy(), loss.numpy()))
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)

        if i % args.i_img == 0:
            # Log a rendered validation view to Tensorboard
            img_i = np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3, :4]
            if args.flag_patch_ray:
                rgb, disp, acc, extras = render_patch(H, W, focal, chunk=int(args.chunk / (args.pr_patch_size*args.pr_patch_size)),
                                                      c2w=pose, pr_patch_size=args.pr_patch_size, **render_kwargs_test)
            else:
                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                **render_kwargs_test)

            psnr = mse2psnr(img2mse(rgb, target))

            # Save out the validation image for Tensorboard-free monitoring
            testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
            if i == 0:
                os.makedirs(testimgdir, exist_ok=True)
            imageio.imwrite(os.path.join(testimgdir, '{:06d}.png'.format(i)), to8b(rgb))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                tf.contrib.summary.image('disp', disp[tf.newaxis, ..., tf.newaxis])
                tf.contrib.summary.image('acc', acc[tf.newaxis, ..., tf.newaxis])
                tf.contrib.summary.scalar('psnr_holdout', psnr)
                tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])

            if args.N_importance > 0:
                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                    tf.contrib.summary.image(
                        'rgb0', to8b(extras['rgb0'])[tf.newaxis])
                    tf.contrib.summary.image(
                        'disp0', extras['disp0'][tf.newaxis, ..., tf.newaxis])
                    tf.contrib.summary.image(
                        'z_std', extras['z_std'][tf.newaxis, ..., tf.newaxis])

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        if i % args.i_video == 0 and i > 0:
            if args.flag_patch_ray:
                rgbs, disps = render_path(
                    render_poses, hwf, args.chunk, render_kwargs_test, pr_patch_size=args.pr_patch_size)
            else:
                rgbs, disps = render_path(
                    render_poses, hwf, args.chunk, render_kwargs_test)

            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4',
                             to8b(disps / np.max(disps)), fps=30, quality=8)

            if args.use_viewdirs:
                render_kwargs_test['c2w_staticcam'] = render_poses[0][:3, :4]
                if args.flag_patch_ray:
                    rgbs_still, _ = render_path(
                        render_poses, hwf, args.chunk, render_kwargs_test, pr_patch_size=args.pr_patch_size)
                else:
                    rgbs_still, _ = render_path(
                        render_poses, hwf, args.chunk, render_kwargs_test)
                render_kwargs_test['c2w_staticcam'] = None
                imageio.mimwrite(moviebase + 'rgb_still.mp4',
                                 to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('TEST: test poses shape', poses[i_test].shape)
            if args.flag_patch_ray:
                render_path(poses[i_test], hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir, iter=i)
            else:
                render_path(poses[i_test], hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test],
                            savedir=testsavedir, iter=i, pr_patch_size=args.pr_patch_size)
            print('TEST: saved test set')


        global_step.assign_add(1)


if __name__ == '__main__':
    train()
