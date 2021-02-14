from easydict import EasyDict

opt = EasyDict(dict(
    load_model='fairmot_dla34.pth',

    K=500, arch='dla_34', batch_size=12, cat_spec_wh=False,
    chunk_sizes=[6, 6], conf_thres=0.4, data_cfg='../src/lib/cfg/data.json',
    data_dir='/data/yfzhang/MOT/JDE', dataset='jde',
    debug_dir='/home/vladbakhteev/projects/people_det/FairMOT/src/lib/../../exp/mot/default/debug',
    dense_wh=False, det_thres=0.3, down_ratio=4,
    exp_dir='/home/vladbakhteev/projects/people_det/FairMOT/src/lib/../../exp/mot',
    exp_id='default', fix_res=True, gpus=[0, 1], gpus_str='0, 1', head_conv=256,
    heads={'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}, hide_data_time=False, hm_weight=1,
    id_loss='ce', id_weight=1, img_size=(1088, 608), input_h=608, input_res=1088,
    input_video='../videos/MOT16-03.mp4', input_w=1088, keep_res=False,
    lr=0.0001, lr_step=[20],
    ltrb=True, master_batch_size=6, mean=[0.408, 0.447, 0.47], metric='loss',
    min_box_area=100, mse_loss=False, nID=14455, nms_thres=0.4, norm_wh=False,
    not_cuda_benchmark=False, not_prefetch_test=False, not_reg_offset=False,
    num_classes=1, num_epochs=30, num_iters=-1, num_stacks=1, num_workers=8,
    off_weight=1, output_format='video', output_h=152, output_res=272,
    output_root='../demos', output_w=272, pad=31, print_iter=0, reg_loss='l1',
    reg_offset=True, reid_dim=128, resume=False,
    root_dir='/home/vladbakhteev/projects/people_det/FairMOT/src/lib/../..',
    save_all=False,
    save_dir='/home/vladbakhteev/projects/people_det/FairMOT/src/lib/../../exp/mot/default',
    seed=317, std=[0.289, 0.274, 0.278], task='mot', test=False, test_hie=False,
    test_mot15=False, test_mot16=False, test_mot17=False, test_mot20=False,
    track_buffer=30, trainval=False, val_hie=False, val_intervals=5, val_mot15=False,
    val_mot16=False, val_mot17=False, val_mot20=False, vis_thresh=0.5, wh_weight=0.1
))