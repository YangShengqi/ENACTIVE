"""primitive maneuver action"""
semantic_maneuver = {
    "maintain": {
        "horizontal_cmd": 0,
        "vertical_cmd": 0,
        "vel_cmd": 680,
        "ny_cmd": 3.5
    },
    "intercept_cata": {
        "horizontal_cmd": 1,
        "vertical_cmd": 1,
        "vel_cmd": 680,
        "ny_cmd": 3.5
    },
    "intercept_level": {
        "horizontal_cmd": 1,
        "vertical_cmd": 0,
        "vel_cmd": 680,
        "ny_cmd": 3.5
    },
    "intercept_climb_20": {
        "horizontal_cmd": 1,
        "vertical_cmd": 5,
        "vel_cmd": 680,
        "ny_cmd": 3.5
    },
    "crank_50": {
        "horizontal_cmd": 3,
        "vertical_cmd": 1,
        "vel_cmd": 680,
        "ny_cmd": 5
    },
    "climb_20": {
        "horizontal_cmd": 1,
        "vertical_cmd": 5,
        "vel_cmd": 680,
        "ny_cmd": 3.5
    },
    "in": {
        "horizontal_cmd": 1,
        "vertical_cmd": 0,
        "vel_cmd": 680,
        "ny_cmd": 3.5
    },
    "out": {
        "horizontal_cmd": 5,
        "vertical_cmd": 0,
        "vel_cmd": 680,
        "ny_cmd": 3.5
    },
    "notch": {
        "horizontal_cmd": 4,
        "vertical_cmd": 0,
        "vel_cmd": 680,
        "ny_cmd": 5
    },
    "abort_no_dive": {
        "horizontal_cmd": 5,
        "vertical_cmd": 0,
        "vel_cmd": 680,
        "ny_cmd": 6
    },
    "abort_dive_25": {
        "horizontal_cmd": 5,
        "vertical_cmd": 3,
        "vel_cmd": 680,
        "ny_cmd": 6
    },
    "split_s": {
        "horizontal_cmd": 5,
        "vertical_cmd": 6,
        "vel_cmd": 680,
        "ny_cmd": 8
    },
    "level_accelerate_escape": {
        "horizontal_cmd": 0,
        "vertical_cmd": 0,
        "vel_cmd": 680,
        "ny_cmd": 5
    }
}

"""top tactics"""
top_tactics = {
    "all_defense": "all_defense",
    "offense_and_defense": "offense_and_defense",
    "all_offense": "all_offense"
}

"""macro tactics"""
macro_tactics = {
    "offense": {
        "intercept_cata": "intercept_cata",
        "intercept_level": "intercept_level",
        "intercept_climb": "intercept_climb",
        "crank_50": "crank_50",
        "climb": "climb"
    },
    "defense": {
        "abort_no_dive": "abort_no_dive",
        "abort_dive_25_1k": "abort_dive_25_1k",
        "abort_dive_25_2k": "abort_dive_25_2k",
        "split_s": "split_s",
        "escape": "escape"
    },
    "other": {
        "maintain": "maintain"
    }
}


class MagicNumber:
    """magic numbers are set here, don't set in function directly"""

    """time(s)"""
    min_threat_missile_tgo = 35  # offense
    min_threat_missile_splits_tgo = 30

    min_crank_time = 15

    """angle(degree)"""
    threshold_abort_delta_ao = 15  # defense
    threshold_splits_delta_chi = 135
    threshold_splits_theta = 20

    threshold_launch_ta = 60  # offense
    threshold_launch_ao = 45
    threshold_face_close_offense_ta = 150
    threshold_tail_close_offense_ta = 30
    threshold_alpha_angle = 120

    """range(m)"""
    # min_skate_range = 65000
    # max_skate_range = 80000
    max_between_bandit_range = 40000  # offense
    min_escape_range = 20000
    upper_far_offense_range = 70000
    lower_far_offense_range = 50000
    lower_mid_offense_range = 40000
    face_close_offense_range = 25000
    close_offense_range = 20000
    tail_close_offense_range = 15000

    border_range = 10000

    """height(m)"""
    mid_splits_height = 5000  # defense
    min_splits_height = 4000
    min_large_abort_dive_25_height = 3000
    min_small_abort_no_dive_height = 2000
    large_abort_dive_25_delta_height = 2000
    small_abort_dive_25_delta_height = 1000

    max_offense_height = 8000  # offense
    min_offense_height = 4000
    min_intercept_climb_20_delta_height = 1000
    min_climb_20_delta_height = 1000

    """v(m/s)"""
    min_threat_missile_tas = 550


