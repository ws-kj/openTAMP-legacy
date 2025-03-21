#for N in 1 2 3 4 5
#do
#for S in third
#do

#python3 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.jnt_pick_hyp \

python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v94_spotnav \
                                                -no 1 -llus 100  -hlus 100 \
                                                -spl -mask -hln 2 -hldim 256 -lldim 256 \
                                                -retime -vel 0.3 -eta 5 -softev \
                                                -lr_schedule adaptive \
						                        -imwidth 64 -imheight 64 \
                                                -obs_del -hist_len 2 -prim_first_wt 20 -lr 0.0002 \
                                                -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
                                                -add_noop 2 --permute_hl 1 \
                                                -expl_wt 10 -expl_eta 4 \
                                                -col_coeff 0.0 \
                                                -motion 5 \
                                                -ntest 1 \
                                                -n_gpu 0 \
                                                -rollout 1 \
                                                -task 1 \
                                                -v \
                                                -render \
                                                -test namo_objs1_1/spotnav_test_0 \
                                                -ind 1 \
                                                -post -pre \
                                                -warm 100 \
                                                -neg_ratio 0. -opt_ratio 0.9 -dagger_ratio 0.1 \
                                                -descr spotnav_test_human & 
# sleep 1800 
# pkill -f run_train -9
# pkill -f ros -9
# sleep 5


#done
#done

