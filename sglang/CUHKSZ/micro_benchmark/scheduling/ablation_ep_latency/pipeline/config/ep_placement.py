gpu_config_4 = [
    {"route_experts": list(range(0, 20)), "redundant_experts": list(range(20, 30))},  # GPU0
    {"route_experts": list(range(20, 40)), "redundant_experts": list(range(40, 50))}, # GPU1
    {"route_experts": list(range(40, 60)), "redundant_experts": list(range(60, 70))}, # GPU2
    {"route_experts": list(range(60, 80)), "redundant_experts": list(range(0, 10))},  # GPU3
]

EXPERT_RANGES_6 = [
    (0, 28), # GPU0
    (29, 57), # GPU1
    (58, 86), # GPU2
    (87, 115), # GPU3
    (116, 144), # GPU4
    (145, 173) # GPU5
]

gpu_config_6 = [
    {"route_experts": list(range(0, 27)),   "redundant_experts": list(range(27, 29))},    # GPU0: 27+2=29
    {"route_experts": list(range(27, 54)),  "redundant_experts": list(range(54, 56))},    # GPU1: 27+2=29
    {"route_experts": list(range(54, 81)),  "redundant_experts": list(range(81, 83))},    # GPU2: 27+2=29
    {"route_experts": list(range(81, 108)), "redundant_experts": list(range(108,110))},   # GPU3: 27+2=29
    {"route_experts": list(range(108,134)), "redundant_experts": list(range(134,137))},   # GPU4: 26+3=29
    {"route_experts": list(range(134,160)), "redundant_experts": list(range(0, 3))},      # GPU5: 26+3=29
]

EXPERT_RANGES_8 = [
    (0, 28), # GPU0
    (29, 57), # GPU1
    (58, 86), # GPU2
    (87, 115), # GPU3
    (116, 144), # GPU4
    (145, 173), # GPU5
    (174, 202), # GPU6
    (203, 231) # GPU7
]

gpu_config_8 = [
    {"route_experts": list(range(0, 20)), "redundant_experts": list(range(20, 29))},  # GPU0
    {"route_experts": list(range(20, 40)), "redundant_experts": list(range(40, 49))}, # GPU1
    {"route_experts": list(range(40, 60)), "redundant_experts": list(range(60, 69))}, # GPU2
    {"route_experts": list(range(60, 80)), "redundant_experts": list(range(80, 89))},  # GPU3
    {"route_experts": list(range(80, 100)), "redundant_experts": list(range(100, 109))},  # GPU4
    {"route_experts": list(range(100, 120)), "redundant_experts": list(range(120, 129))},  # GPU5
    {"route_experts": list(range(120, 140)), "redundant_experts": list(range(140, 149))},  # GPU6
    {"route_experts": list(range(140, 160)), "redundant_experts": list(range(0, 9))},  # GPU7        
]

EXPERT_RANGES_10= [
    (0, 28), # GPU0
    (29, 57), # GPU1
    (58, 86), # GPU2
    (87, 115), # GPU3
    (116, 144), # GPU4
    (145, 173), # GPU5
    (174, 202), # GPU6
    (203, 231), # GPU7
    (232, 260), # GPU8
    (261, 289), # GPU9
]

gpu_config_10 = [
    {"route_experts": list(range(0, 16)), "redundant_experts": list(range(16, 29))},  # GPU0
    {"route_experts": list(range(16, 32)), "redundant_experts": list(range(32, 45))}, # GPU1
    {"route_experts": list(range(32, 48)), "redundant_experts": list(range(48, 61))}, # GPU2
    {"route_experts": list(range(48, 64)), "redundant_experts": list(range(64, 77))}, # GPU3
    {"route_experts": list(range(64, 80)), "redundant_experts": list(range(80, 93))}, # GPU4
    {"route_experts": list(range(80, 96)), "redundant_experts": list(range(96, 109))}, # GPU5
    {"route_experts": list(range(96, 112)), "redundant_experts": list(range(112, 125))}, # GPU6
    {"route_experts": list(range(112, 128)), "redundant_experts": list(range(128, 141))}, # GPU7
    {"route_experts": list(range(128, 144)), "redundant_experts": list(range(144, 157))}, # GPU8
    {"route_experts": list(range(144, 160)), "redundant_experts": list(range(0, 13))},  # GPU9
]

EXPERT_RANGES_12= [
    (0, 28), # GPU0
    (29, 57), # GPU1
    (58, 86), # GPU2
    (87, 115), # GPU3
    (116, 144), # GPU4
    (145, 173), # GPU5
    (174, 202), # GPU6
    (203, 231), # GPU7
    (232, 260), # GPU8
    (261, 289), # GPU9
    (290, 318), # GPU10
    (319, 347), # GPU11
]

gpu_config_12 = [
    # 每个gpu上29个expert，前4个gpu 每个14个route，15个redundant。后8个gpu 每个13个route，16个redundant。
    {"route_experts": list(range(0, 14)), "redundant_experts": list(range(14, 29))},  # GPU0
    {"route_experts": list(range(14, 28)), "redundant_experts": list(range(29, 44))}, # GPU1
    {"route_experts": list(range(28, 42)), "redundant_experts": list(range(44, 59))}, # GPU2
    {"route_experts": list(range(42, 56)), "redundant_experts": list(range(59, 74))}, # GPU3
    {"route_experts": list(range(56, 69)), "redundant_experts": list(range(74, 90))}, # GPU4
    {"route_experts": list(range(69, 82)), "redundant_experts": list(range(90, 106))}, # GPU5
    {"route_experts": list(range(82, 95)), "redundant_experts": list(range(106, 122))}, # GPU6
    {"route_experts": list(range(95, 108)), "redundant_experts": list(range(122, 138))}, # GPU7
    {"route_experts": list(range(108, 121)), "redundant_experts": list(range(138, 154))}, # GPU8
    {"route_experts": list(range(121, 134)), "redundant_experts": list(range(0, 16))}, # GPU9
    {"route_experts": list(range(134, 147)), "redundant_experts": list(range(16, 32))}, # GPU10
    {"route_experts": list(range(147, 160)), "redundant_experts": list(range(32, 48))}, # GPU11    
]

EXPERT_RANGES_14= [
    (0, 28), # GPU0
    (29, 57), # GPU1
    (58, 86), # GPU2
    (87, 115), # GPU3
    (116, 144), # GPU4
    (145, 173), # GPU5
    (174, 202), # GPU6
    (203, 231), # GPU7
    (232, 260), # GPU8
    (261, 289), # GPU9
    (290, 318), # GPU10
    (319, 347), # GPU11
    (348, 376), # GPU12
    (377, 405), # GPU13
]

gpu_config_14 = [
    {"route_experts": list(range(0, 11)),    "redundant_experts": list(range(11, 29))},    # GPU0: 11+18
    {"route_experts": list(range(11, 22)),   "redundant_experts": list(range(29, 47))},    # GPU1: 11+18
    {"route_experts": list(range(22, 33)),   "redundant_experts": list(range(47, 65))},    # GPU2: 11+18
    {"route_experts": list(range(33, 44)),   "redundant_experts": list(range(65, 83))},    # GPU3: 11+18
    {"route_experts": list(range(44, 55)),   "redundant_experts": list(range(83, 101))},   # GPU4: 11+18
    {"route_experts": list(range(55, 66)),   "redundant_experts": list(range(101,119))},   # GPU5: 11+18
    {"route_experts": list(range(66, 77)),   "redundant_experts": list(range(119,137))},   # GPU6: 11+18
    {"route_experts": list(range(77, 88)),   "redundant_experts": list(range(137,155))},   # GPU7: 11+18
    {"route_experts": list(range(88, 100)),  "redundant_experts": list(range(0, 17))},     # GPU8: 12+17
    {"route_experts": list(range(100,112)),  "redundant_experts": list(range(17, 34))},    # GPU9: 12+17
    {"route_experts": list(range(112,124)),  "redundant_experts": list(range(34, 51))},    # GPU10:12+17
    {"route_experts": list(range(124,136)),  "redundant_experts": list(range(51, 68))},    # GPU11:12+17
    {"route_experts": list(range(136,148)),  "redundant_experts": list(range(68, 85))},    # GPU12:12+17
    {"route_experts": list(range(148,160)),  "redundant_experts": list(range(85,102))},    # GPU13:12+17
]

EXPERT_RANGES_16= [
    (0, 28), # GPU0
    (29, 57), # GPU1
    (58, 86), # GPU2
    (87, 115), # GPU3
    (116, 144), # GPU4
    (145, 173), # GPU5
    (174, 202), # GPU6
    (203, 231), # GPU7
    (232, 260), # GPU8
    (261, 289), # GPU9
    (290, 318), # GPU10
    (319, 347), # GPU11
    (348, 376), # GPU12
    (377, 405), # GPU13
    (406, 434), # GPU14
    (435, 463), # GPU15
]

gpu_config_16 = [
    {"route_experts": list(range(0, 10)),   "redundant_experts": list(range(10, 29))},   # GPU0
    {"route_experts": list(range(10, 20)),  "redundant_experts": list(range(29, 48))},   # GPU1
    {"route_experts": list(range(20, 30)),  "redundant_experts": list(range(48, 67))},   # GPU2
    {"route_experts": list(range(30, 40)),  "redundant_experts": list(range(67, 86))},   # GPU3
    {"route_experts": list(range(40, 50)),  "redundant_experts": list(range(86, 105))},  # GPU4
    {"route_experts": list(range(50, 60)),  "redundant_experts": list(range(105,124))},  # GPU5
    {"route_experts": list(range(60, 70)),  "redundant_experts": list(range(124,143))},  # GPU6
    {"route_experts": list(range(70, 80)),  "redundant_experts": list(range(143,160))},  # GPU7
    {"route_experts": list(range(80, 90)),  "redundant_experts": list(range(0,19))},     # GPU8
    {"route_experts": list(range(90,100)),  "redundant_experts": list(range(19,38))},    # GPU9
    {"route_experts": list(range(100,110)), "redundant_experts": list(range(38,57))},    # GPU10
    {"route_experts": list(range(110,120)), "redundant_experts": list(range(57,76))},    # GPU11
    {"route_experts": list(range(120,130)), "redundant_experts": list(range(76,95))},    # GPU12
    {"route_experts": list(range(130,140)), "redundant_experts": list(range(95,114))},   # GPU13
    {"route_experts": list(range(140,150)), "redundant_experts": list(range(114,133))},  # GPU14
    {"route_experts": list(range(150,160)), "redundant_experts": list(range(133,152))},  # GPU15
]
    
