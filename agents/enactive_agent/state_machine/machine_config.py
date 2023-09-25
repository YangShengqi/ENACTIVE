"""primitive maneuver action"""
semantic_maneuver_default = {
    "maintain": {
        "horizontal_cmd": 0,
        "vertical_cmd": 0,
        "vel_cmd": 680,
        "ny_cmd": 5,
        "clockwise_cmd": 0,
        "base_direction": 10
    },
    "climb_20": {
        "horizontal_cmd": 1,
        "vertical_cmd": 5,
        "vel_cmd": 680,
        "ny_cmd": 5,
        "clockwise_cmd": 0,
        "base_direction": 10
    },

    # intercept #
    "intercept_cata": {
        "horizontal_cmd": 1,
        "vertical_cmd": 1,
        "vel_cmd": 680,
        "ny_cmd": 5,
        "clockwise_cmd": 0,
        "base_direction": 10
    },
    "intercept_level": {
        "horizontal_cmd": 1,
        "vertical_cmd": 0,
        "vel_cmd": 680,
        "ny_cmd": 5,
        "clockwise_cmd": 0,
        "base_direction": 10
    },
    "intercept_climb_20": {
        "horizontal_cmd": 1,
        "vertical_cmd": 5,
        "vel_cmd": 680,
        "ny_cmd": 5,
        "clockwise_cmd": 0,
        "base_direction": 10
    },

    # crank #
    "in": {
        "horizontal_cmd": 1,
        "vertical_cmd": 1,
        "vel_cmd": 680,
        "ny_cmd": 5,
        "clockwise_cmd": 0,
        "base_direction": 10
    },
    "crank_30": {
        "horizontal_cmd": 2,
        "vertical_cmd": 1,
        "vel_cmd": 680,
        "ny_cmd": 5,
        "clockwise_cmd": 0,
        "base_direction": 10
    },
    "crank_50": {
        "horizontal_cmd": 3,
        "vertical_cmd": 1,
        "vel_cmd": 680,
        "ny_cmd": 5,
        "clockwise_cmd": 0,
        "base_direction": 10
    },
    "crank_90": {
        "horizontal_cmd": 4,
        "vertical_cmd": 0,
        "vel_cmd": 680,
        "ny_cmd": 6,
        "clockwise_cmd": 0,
        "base_direction": 10
    },
    "out_120": {
        "horizontal_cmd": 6,
        "vertical_cmd": 0,
        "vel_cmd": 680,
        "ny_cmd": 6,
        "clockwise_cmd": 0,
        "base_direction": 10
    },
    "out_150": {
        "horizontal_cmd": 7,
        "vertical_cmd": 0,
        "vel_cmd": 680,
        "ny_cmd": 6,
        "clockwise_cmd": 0,
        "base_direction": 10
    },
    "out": {
        "horizontal_cmd": 5,
        "vertical_cmd": 0,
        "vel_cmd": 680,
        "ny_cmd": 6,
        "clockwise_cmd": 0,
        "base_direction": 10
    },
    "abort_dive_25": {
        "horizontal_cmd": 5,
        "vertical_cmd": 3,
        "vel_cmd": 680,
        "ny_cmd": 6,
        "clockwise_cmd": 0,
        "base_direction": 10
    },

    # evade #
    "split_s": {
        "horizontal_cmd": 5,
        "vertical_cmd": 6,
        "vel_cmd": 680,
        "ny_cmd": 8,
        "clockwise_cmd": 0,
        "base_direction": 10
    }
}

"""top tactics"""
top_tactics = {
    "all_defense": "all_defense",
    "all_offense": "all_offense"
}

"""macro tactics"""
macro_tactics = {
    "offense": {
        "intercept_cata": "intercept_cata",
        "intercept_level": "intercept_level",
        "intercept_climb": "intercept_climb",
        "climb": "climb",
        "banzai": "banzai"
    },
    "defense": {
        "abort": {
            "abort": "abort",
            "abort_dive_25_1k": "abort_dive_25_1k",
            "abort_dive_25_2k": "abort_dive_25_2k",
            "abort_dive_25_3k": "abort_dive_25_3k",
            "split_s": "split_s"
        },
        "evade": {
            "crank_30": "crank_30",
            "crank_50": "crank_50",
            "crank_50_dive_25": "crank_50_dive_25",
            "notch_dive_25": "notch_dive_25",
            "circle": "circle",
            "circle_dive_25": "circle_dive_25",
            "snake_50": "snake_50",
            "snake_50_dive_25": "snake_50_dive_25",
            "snake_90": "snake_90",
            "snake_90_dive_25": "snake_90_dive_25",
            "change_direction": "change_direction"
        }
    },
    "other": {
        "maintain": "maintain"
    }
}

threat_level = {
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high"
}


class MagicNumber:
    """magic numbers are set here, don't set in function directly"""

    """time(s)"""
    mid_threat_level_msl_tgo = 20
    # high_threat_level_msl_tgo = 22
    # turn_in_time = 10
    min_lost_guide_time = 10

    min_crank_time = 5
    min_notch_time = 15
    # snake_50_turn_time = 15
    snake_50_maintain_time = 15
    snake_90_maintain_time_0 = 6
    snake_90_maintain_time_1 = 10
    snake_clockwise_counter = 4
    banzai_shoot_interval = 10

    """angle(degree)"""
    threshold_abort_delta_ao = 10  # defense
    threshold_splits_delta_chi = 135
    threshold_splits_theta = 20

    threshold_launch_ta = 60  # offense
    threshold_launch_ao = 30
    threshold_face_close_offense_ta = 150
    threshold_tail_close_offense_ta = 90
    threshold_alpha_angle = 120

    """range(m)"""
    # min_skate_range = 65000
    # max_skate_range = 80000
    max_between_bandit_range = 40000  # offense
    min_team_range = 15000
    min_escape_range = 20000
    min_safe_range = 45000
    min_png_delta_range = 4000
    # upper_far_offense_range = 80000
    # lower_far_offense_range = 60000
    # lower_mid_offense_range = 45000
    # face_close_offense_range = 35000
    # close_offense_range = 28000
    # tail_close_offense_range = 18000

    high_threat_level_msl_range = 30000
    mid_threat_level_msl_range = 45000
    far_retreat_range = 70000
    close_retreat_range = 60000
    threshold_threat_msl_launch_range = 55000

    far_border_range = 30000
    mid_border_range = 20000
    near_border_range = 10000

    """height(m)"""
    mid_splits_height = 5000  # defense
    min_splits_height = 4000
    min_large_abort_dive_25_height = 3000
    min_small_abort_no_dive_height = 2000
    abort_dive_25_3k_delta_height = 3000
    abort_dive_25_2k_delta_height = 2000
    abort_dive_25_1k_delta_height = 1000

    max_offense_height = 8000  # offense
    min_offense_height = 5000
    min_intercept_climb_20_delta_height = 3000
    min_climb_20_delta_height = 2000

    """v(m/s)"""
    min_threat_missile_tas = 600
    min_missile_r_dot = 200


