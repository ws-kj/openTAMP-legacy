#for N in 1 2 3 4 5
#do
#for S in third
#do

#python3 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.jnt_pick_hyp \

TRIAL_NAME='imitation_test'


#python3 -W ignore policy_hooks/run_training.py -c policy_hooks.new_specs.floorplan_domain_belief \
python3 -W ignore policy_hooks/run_training.py -c new_specs.will_test_domain.new_env_hyperparam \
                                                -no 1 \
                                                -llus 100  -hlus 100 \
                                                -spl \
                                                -hln 4 -lln 4 -hldim 128 -lldim 128 \
                                                -vel 0.3 -eta 5 -softev \
                                                -lr_schedule adaptive \
                                                -imwidth 256 -imheight 256 \
                                                -hist_len 2 \
                                                -prim_first_wt 20 -lr 0.00005 \
                                                -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
                                                -add_noop 2 --permute_hl 1 \
                                                -expl_wt 10 -expl_eta 4 \
                                                -col_coeff 0.0 \
                                                -motion 4 \
                                                -n_gpu 1 \
                                                -run_time 1800 \
                                                -hl_recur \
                                                -rollout 1 \
                                                -task 1 \
                                                -post -pre \
                                                -warm 100 \
                                                -neg_ratio 0. -opt_ratio 0.9 -dagger_ratio 0.1 \
                                                -ind 0 \
						-descr server_floorplan_nav_belief_pointerobs_fulllength_taskmap_smallbuf_addobsvisibilitydebug_splitnet_sweepskolemupdate_truetargpred_30minrun \
#                                                -test namo_objs1_1/server_floorplan_nav_belief_pointerobs_fulllength_taskmap_smallbuf_addobsvisibilitydebug_splitnet_sweepskolemupdate_truetargpred_30minrun_0

# sleep 1800 
# pkill -f run_train -9
# pkill -f ros -9
# sleep 5∂


#done
#done

