mkdir -p output

# RL4RS environment

mkdir -p output/KuaiRec/
mkdir -p output/KuaiRec/agents/

output_path="output/KuaiRec/"
log_name="user_KRMBUserResponse_lr0.0001_reg0_nlayer2"

# environment args
ENV_CLASS='KREnvironment_WholeSession_GPU'
# ENV_CLASS='KREnvironment_WholeSession_TemperDiscount'
MAX_STEP=30
SLATE_SIZE=10 #6
EP_BS=64 # 64
RHO=0.2
TEMPER_DISCOUNT=2.0

# policy args
POLICY_CLASS='OneStagePolicy_LHP1_d1'
HA_VAR=0.1
HA_CLIP=1.0
# if explore the effect action set --policy_do_effect_action_explore

# critic args
CRITIC_CLASS='LVCritic1'


# buffer args
BUFFER_CLASS='HRLBuffer1'
BUFFER_SIZE=2500

# agent args
AGENT_CLASS='LHP1'
GAMMA=0.9
REWARD_FUNC='get_immediate_reward'
N_ITER=20000
INITEP=0.01
ELBOW=0.1
EXPLORE_RATE=1.0  #1.0
BS=128
# if want to explore in train set --do_explore_in_train
#MEG='econd-0.3_f0.1d(1+up)_wmlp_tau0.01_varlx'
# MEG='trans'
# REW='E2'
for HA_VAR in 0.1
do
    for REG in 0.00001
    do
        for INITEP in 0.01
        do
            for CRITIC_LR in 0.001
            do
                for ACTOR_LR in 0.0001 # 0.0001, 0.00001
                do
                    for SEED in 11 15 19 #17 19 23 # 11 13 15 # 
                    do
                        file_key=paper1/para/lambda/${MAX_STEP}/5_${AGENT_CLASS}_${SEED}
                        # file_key=paper1/new/${MAX_STEP}/d_${AGENT_CLASS}_${SEED}
                        mkdir -p ${output_path}agents/${file_key}/
                        python train_actor_critic.py\
                            --env_class ${ENV_CLASS}\
                            --policy_class ${POLICY_CLASS}\
                            --critic_class ${CRITIC_CLASS}\
                            --buffer_class ${BUFFER_CLASS}\
                            --agent_class ${AGENT_CLASS}\
                            --seed ${SEED}\
                            --cuda 1\
                            --max_step_per_episode ${MAX_STEP}\
                            --initial_temper ${MAX_STEP}\
                            --uirm_log_path ${output_path}env/log/${log_name}.model.log\
                            --slate_size ${SLATE_SIZE}\
                            --episode_batch_size ${EP_BS}\
                            --item_correlation ${RHO}\
                            --temper_discount ${TEMPER_DISCOUNT}\
                            --single_response\
                            --policy_action_hidden 256 64\
                            --policy_noise_var ${HA_VAR}\
                            --policy_noise_clip ${HA_CLIP}\
                            --state_user_latent_dim 16\
                            --state_item_latent_dim 16\
                            --state_transformer_enc_dim 32\
                            --state_transformer_n_head 4\
                            --state_transformer_d_forward 64\
                            --state_transformer_n_layer 3\
                            --state_dropout_rate 0.1\
                            --critic_hidden_dims 256 64\
                            --critic_dropout_rate 0.1\
                            --buffer_size ${BUFFER_SIZE}\
                            --gamma ${GAMMA}\
                            --reward_func ${REWARD_FUNC}\
                            --n_iter ${N_ITER}\
                            --train_every_n_step 20\
                            --initial_epsilon ${INITEP}\
                            --final_epsilon ${INITEP}\
                            --elbow_epsilon ${ELBOW}\
                            --explore_rate ${EXPLORE_RATE}\
                            --check_episode 10\
                            --save_episode 200\
                            --save_path ${output_path}agents/${file_key}/model\
                            --actor_lr ${ACTOR_LR}\
                            --actor_decay ${REG}\
                            --batch_size ${BS}\
                            --critic_lr ${CRITIC_LR}\
                            --critic_decay ${REG}\
                            --target_mitigate_coef 0.1\
                            > ${output_path}agents/${file_key}/log
                    done
                done
            done
        done
    done
done
