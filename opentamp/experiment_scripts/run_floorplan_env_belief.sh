#for N in 1 2 3 4 5
#do
#for S in third
#do

#python3 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.jnt_pick_hyp \

# runs the main job

TRIAL_NAME='floorplan_nav_belief_singleroom'

python3 -W ignore policy_hooks/run_training.py -c new_specs.floorplan_domain_belief.new_env_hyperparam \
                                                -llus 10  -hlus 10 \
                                                -spl -hln 2 -lln 4 -hldim 32 -lldim 128 \
                                                -lr_schedule fixed \
                                                -imwidth 64 -imheight 64 -imchannels 3 \
                                                -lr 0.0001 -hllr 0.001 -contlr 0.0001 -lldec 0.0001 -hldec 0.0001 \
                                                -motion 1 \
                                                -hl_recur \
                                                -n_gpu 1 \
                                                -num_test 1 \
                                                -rollout 1 \
                                                -task 1 \
                                                -batch 32 \
                                                -warm 100 \
                                                -run_time 3600 \
                                                -opt_ratio 1.0 -dagger_ratio 0.0 \
						-descr $TRIAL_NAME \
#                                                -plan_only \
                                                -test \
                                                -render 

# adds renders after job is done -- majority of options here ignored

# python3 -W ignore policy_hooks/run_training.py -c new_specs.floorplan_domain_belief.new_env_hyperparam \
#                                                 -no 1 -llus 100  -hlus 100 \
#                                                 -spl -mask -hln 2 -hldim 256 -lldim 256 \
#                                                 -retime -vel 0.3 -eta 5 -softev \
#                                                 -lr_schedule adaptive \
#                                                 -imwidth 64 -imheight 64 \
#                                                 -hist_len 2 -prim_first_wt 20 -lr 0.00005 \
#                                                 -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
#                                                 -add_noop 2 --permute_hl 1 \
#                                                 -expl_wt 10 -expl_eta 4 \
#                                                 -col_coeff 0.0 \
#                                                 -motion 4 \
#                                                 -n_gpu 0 \
#                                                 -rollout 0 \
#                                                 -task 1 \
#                                                 -post -pre \
#                                                 -warm 100 \
#                                                 -neg_ratio 0. -opt_ratio 0.9 -dagger_ratio 0.1 \
#                                                 -ind 0 \
# 						-descr $TRIAL_NAME \
#                                                 -test namo_objs1_1/${TRIAL_NAME}_0 \
#                                                 -render

# pkill -f ros -9
# sleep 5


#done
#done

