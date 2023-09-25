# some expert made reward hyperparamers #
# from train.config import Config

# env = Config.env
# env.reset()

# each aircraft have following events #
# crash_event, death_event, in_border_event, out_border_event,
# lock_event, unlock_event, lost_event, shoot_down_event, be_shoot_down_event, stall_event
# each missile have following events #
# fire_event, hit_event, miss_event

# usually use following events to create reward #
# 1. crash_event 2. be_shoot_down_event 3. stall_event 4.death_event
# 5. in_border_event 6. out_border_event
# 7. shoot_down_event 8.fire_event

radical_reward_parameters = dict(crash_extra_reward=-100, be_shoot_down_extra_reward=0,
                                 stall_extra_reward=-100, death_reward=-400,
                                 in_border_reward=50, out_border_reward=-50,
                                 shoot_down_reward=600, fire_reward=-50, accumulate_fire_reward=-30,
                                 accumulate_shoot_down_reward=200,
                                 accumulate_death_reward=-200,
                                 all_shoot_down_event_reward=500)

conservative_reward_parameters = dict(crash_extra_reward=-100, be_shoot_down_extra_reward=0,
                                      stall_extra_reward=-100, death_reward=-600,
                                      in_border_reward=50, out_border_reward=-50,
                                      shoot_down_reward=400, fire_reward=-50, accumulate_fire_reward=-30,
                                      accumulate_shoot_down_reward=200,
                                      accumulate_death_reward=-200,
                                      all_shoot_down_event_reward=500)

origin_reward_parameters = dict(crash_extra_reward=-100, be_shoot_down_extra_reward=0,
                                stall_extra_reward=-100, death_reward=-500,
                                in_border_reward=50, out_border_reward=-50,
                                shoot_down_reward=500, fire_reward=-50, accumulate_fire_reward=-30,
                                accumulate_shoot_down_reward=200,
                                accumulate_death_reward=-200,
                                all_shoot_down_event_reward=500)

reward_parameters = [origin_reward_parameters,
                     radical_reward_parameters,
                     conservative_reward_parameters]
