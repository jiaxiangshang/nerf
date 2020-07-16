#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jiaxiang Shang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: jiaxiang.shang@gmail.com
@time: 6/17/20 7:53 PM
@desc:
'''
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import time

#
from skimage.metrics import structural_similarity as ssim

#
from tf_viewSyn.nerf.run_nerf_helpers import *
from baselib_python.Common.VisualError import pixel_error_heatmap

def batchify_range(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network_range(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""

    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)
    if viewdirs is not None:
        #input_dirs = tf.broadcast_to(viewdirs[:, None, :], inputs.shape)
        input_dirs = tf.tile(viewdirs[:, None, :], [1, inputs.shape[1], 1])
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = tf.concat([embedded, embedded_dirs], -1)

    outputs_flat = batchify_range(fn, netchunk)(embedded)
    outputs = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render_rays_range(ray_batch, network_fn, network_query_fn,
                      N_samples, retraw=False, lindisp=False, perturb=0., N_importance=0, network_fine=None,
                      white_bkgd=False, raw_noise_std=0., verbose=False, flag_infinite_last=True, embed_fn=None, embed_fn_fine=None):
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    def raw2outputs(raw, z_vals, rays_d, coarse=None, z_vals_next=None):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """

        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu):
            return 1.0 - tf.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        if flag_infinite_last:
            dists = tf.concat([dists, tf.broadcast_to([1e10], dists[..., :1].shape)], axis=-1)  # [N_rays, N_samples]
        else:
            #dists = tf.concat([dists, dists[:, -1:]], axis=-1)  # [N_rays, N_samples]
            dists = tf.concat([dists, tf.broadcast_to([1], dists[..., :1].shape)], axis=-1)  # [N_rays, N_samples]
            # if z_vals_next is not None:
            #     dists = dists - dists[:, -1:]/2.0

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Extract RGB of each sample position along each ray.
        rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        tf.debugging.check_numerics(rgb, 'raw2outputs {}'.format('rgb'))

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        tf.debugging.check_numerics(alpha, 'raw2outputs {}'.format('alpha'))

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * tf.math.cumprod(1. - alpha + 1e-10, axis=-1, exclusive=True)
        tf.debugging.check_numerics(weights, 'raw2outputs {}'.format('weights'))

        # Computed weighted color of each sample along each ray.
        rgb_map = tf.reduce_sum(weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance.
        if z_vals_next is not None:
            z_vals_mid = (z_vals+z_vals_next)/2.0
            depth_map = tf.reduce_sum(weights * z_vals_mid, axis=-1)
        else:
            depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Disparity map is inverse depth.
        disp_map = 1. / tf.maximum(1e-10, depth_map /tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map, alpha

    ###############################
    # batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = tf.linspace(0., 1., N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])
    #z_vals_next = tf.concat([z_vals[:, 1:], tf.broadcast_to([1e10], z_vals[..., :1].shape)], axis=1)
    #z_vals_next = tf.concat([z_vals[:, 1:], z_vals[:, -1:]+tf.abs(z_vals[:, -1:]-z_vals[:, -2:-1])], axis=1)
    z_vals_next = tf.concat([z_vals[:, 1:], z_vals[:, -1:] + 1], axis=1)

    # Perturb sampling time along each ray.
    if 0:
    #if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    #z_val_range = tf.stack([z_vals, z_vals_next], axis=2)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    pts_next = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_next[..., :, None]  # [N_rays, N_samples, 3]
    pts_range = tf.concat([pts, pts_next], axis=2)

    # Evaluate model at each point.
    raw = network_query_fn(pts_range, viewdirs, network_fn, embed_fn)  # [N_rays, N_samples, 4]
    rgb_map, disp_map, acc_map, weights, depth_map_coarse, alpha = raw2outputs(raw, z_vals, rays_d, coarse=True, z_vals_next=z_vals_next)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0, alpha_0 = rgb_map, disp_map, acc_map, alpha

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)

        if 0:
            # Obtain additional integration times to evaluate based on the weights
            # assigned to colors in the coarse model.
            z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
        else:
            if 1:
                #idx = tf.argmax(weights, axis=1)
                idx_pred = tf.nn.softmax(weights*0.1, dim=1)
                # rang = tf.linspace(0., N_samples+1, N_samples)
                # rang = tf.broadcast_to(rang[None, :], [N_rays, N_samples])
                #
                # idx = tf.reduce_sum(tf.multiply(rang, idx_pred), axis=1)
                # idx = tf.cast(idx, tf.int32)

                z_st = tf.reduce_sum(tf.multiply(z_vals, idx_pred), axis=1)
                z_end = tf.reduce_sum(tf.multiply(z_vals_next, idx_pred), axis=1)
                z_st = z_st[:, None]
                z_end = z_end[:, None]
            else:
                z_st = depth_map_coarse[:, None] - tf.abs(far - near) / N_samples / 2.0
                z_end = z_st + tf.abs(far - near) / N_samples

            fine_t_vals = tf.linspace(0., 1.0, N_importance)
            if not lindisp:
                # Space integration times linearly between 'near' and 'far'. Same
                # integration points will be used for all rays.
                z_vals_level1 = z_st * (1. - fine_t_vals) + z_end * (fine_t_vals)
            else:
                # Sample linearly in inverse depth (disparity).
                z_vals_level1 = 1. / (1. / (z_st+1e-6) * (1. - fine_t_vals) + 1. / (z_end+1e-6) * (fine_t_vals) +1e-6)
            z_vals_level1 = tf.broadcast_to(z_vals_level1, [N_rays, N_importance])

            #z_vals = tf.sort(tf.concat([z_vals, z_vals_level1], -1), -1)
            z_vals = tf.sort(z_vals_level1, -1)



        # Obtain all points to evaluate color, density at.
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
                                     z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
        tf.debugging.check_numerics(z_vals, 'output {}'.format('z_vals'))
        tf.debugging.check_numerics(pts, 'output {}'.format('pts'))

        # Make predictions with network_fine.
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn, embed_fn_fine)
        rgb_map, disp_map, acc_map, weights, depth_map_fine, alpha = raw2outputs(
            raw, z_vals, rays_d, coarse=False)

        # depth_map_final = depth_map_coarse + depth_map_fine
        disp_map_final = 1. / tf.maximum(1e-10, depth_map_fine / tf.reduce_sum(weights, axis=-1))

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map_final, 'acc_map': acc_map, 'alpha': alpha}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['alpha_0'] = alpha_0
        ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret

def batchify_rays_range(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays_range(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render_range(H, W, focal,
           chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None, flag_infinite_last=True,
           **kwargs):
    """Render rays

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    near, far = near * \
                tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)

    # Render and reshape
    all_ret = batchify_rays_range(rays, chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def create_nerf_range(args):
    """Instantiate NeRF's MLP model."""

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, 6)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]
    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    grad_vars = model.trainable_variables
    models = {'model': model}

    model_fine = None
    if args.N_importance > 0:
        embed_fn_fine, input_ch_fine = get_embedder(args.multires, args.i_embed, 3)

        model_fine = init_nerf_model(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch_fine, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        grad_vars += model_fine.trainable_variables
        models['model_fine'] = model_fine

    def network_query_fn(inputs, viewdirs, network_fn, embed_fn):
        return run_network_range(
            inputs, viewdirs, network_fn,
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'embed_fn_fine': embed_fn_fine,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'embed_fn': embed_fn,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models