"""https://github.com/NVIDIAGameWorks/NVIDIAImageScaling
"""

import struct

# coef_scale_fp16 from NIS_Config.h (64x8)
coef_scale_fp16 = [
	[0, 0, 15360, 0, 0, 0, 0, 0],
	[6640, 41601, 15360, 8898, 39671, 0, 0, 0],
	[7796, 42592, 15357, 9955, 40695, 0, 0, 0],
	[8321, 43167, 15351, 10576, 41286, 4121, 0, 0],
	[8702, 43537, 15346, 11058, 41797, 4121, 0, 0],
	[9029, 43871, 15339, 11408, 42146, 4121, 0, 0],
	[9280, 44112, 15328, 11672, 42402, 5145, 0, 0],
	[9411, 44256, 15316, 11944, 42690, 5669, 0, 0],
	[9535, 44401, 15304, 12216, 42979, 6169, 0, 0],
	[9667, 44528, 15288, 12396, 43137, 6378, 0, 0],
	[9758, 44656, 15273, 12540, 43282, 6640, 0, 0],
	[9857, 44768, 15255, 12688, 43423, 6903, 0, 0],
	[9922, 44872, 15235, 12844, 43583, 7297, 0, 0],
	[10014, 44959, 15213, 13000, 43744, 7429, 0, 0],
	[10079, 45048, 15190, 13156, 43888, 7691, 0, 0],
	[10112, 45092, 15167, 13316, 44040, 7796, 0, 0],
	[10178, 45124, 15140, 13398, 44120, 8058, 0, 0],
	[10211, 45152, 15112, 13482, 44201, 8256, 0, 0],
	[10211, 45180, 15085, 13566, 44279, 8387, 0, 0],
	[10242, 45200, 15054, 13652, 44360, 8518, 0, 0],
	[10242, 45216, 15023, 13738, 44440, 8636, 0, 0],
	[10242, 45228, 14990, 13826, 44520, 8767, 0, 0],
	[10242, 45236, 14955, 13912, 44592, 8964, 0, 0],
	[10211, 45244, 14921, 14002, 44673, 9082, 0, 0],
	[10178, 45244, 14885, 14090, 44745, 9213, 0, 0],
	[10145, 45244, 14849, 14178, 44817, 9280, 0, 0],
	[10112, 45236, 14810, 14266, 44887, 9378, 0, 0],
	[10079, 45228, 14770, 14346, 44953, 9437, 0, 0],
	[10014, 45216, 14731, 14390, 45017, 9503, 0, 0],
	[9981, 45204, 14689, 14434, 45064, 9601, 0, 0],
	[9922, 45188, 14649, 14478, 45096, 9667, 0, 0],
	[9857, 45168, 14607, 14521, 45120, 9726, 0, 0],
	[9791, 45144, 14564, 14564, 45144, 9791, 0, 0],
	[9726, 45120, 14521, 14607, 45168, 9857, 0, 0],
	[9667, 45096, 14478, 14649, 45188, 9922, 0, 0],
	[9601, 45064, 14434, 14689, 45204, 9981, 0, 0],
	[9503, 45017, 14390, 14731, 45216, 10014, 0, 0],
	[9437, 44953, 14346, 14770, 45228, 10079, 0, 0],
	[9378, 44887, 14266, 14810, 45236, 10112, 0, 0],
	[9280, 44817, 14178, 14849, 45244, 10145, 0, 0],
	[9213, 44745, 14090, 14885, 45244, 10178, 0, 0],
	[9082, 44673, 14002, 14921, 45244, 10211, 0, 0],
	[8964, 44592, 13912, 14955, 45236, 10242, 0, 0],
	[8767, 44520, 13826, 14990, 45228, 10242, 0, 0],
	[8636, 44440, 13738, 15023, 45216, 10242, 0, 0],
	[8518, 44360, 13652, 15054, 45200, 10242, 0, 0],
	[8387, 44279, 13566, 15085, 45180, 10211, 0, 0],
	[8256, 44201, 13482, 15112, 45152, 10211, 0, 0],
	[8058, 44120, 13398, 15140, 45124, 10178, 0, 0],
	[7796, 44040, 13316, 15167, 45092, 10112, 0, 0],
	[7691, 43888, 13156, 15190, 45048, 10079, 0, 0],
	[7429, 43744, 13000, 15213, 44959, 10014, 0, 0],
	[7297, 43583, 12844, 15235, 44872, 9922, 0, 0],
	[6903, 43423, 12688, 15255, 44768, 9857, 0, 0],
	[6640, 43282, 12540, 15273, 44656, 9758, 0, 0],
	[6378, 43137, 12396, 15288, 44528, 9667, 0, 0],
	[6169, 42979, 12216, 15304, 44401, 9535, 0, 0],
	[5669, 42690, 11944, 15316, 44256, 9411, 0, 0],
	[5145, 42402, 11672, 15328, 44112, 9280, 0, 0],
	[4121, 42146, 11408, 15339, 43871, 9029, 0, 0],
	[4121, 41797, 11058, 15346, 43537, 8702, 0, 0],
	[4121, 41286, 10576, 15351, 43167, 8321, 0, 0],
	[0, 40695, 9955, 15357, 42592, 7796, 0, 0],
	[0, 39671, 8898, 15360, 41601, 6640, 0, 0],
]

coef_usm_fp16 = [
	[0, 47309, 15565, 47309, 0, 0, 0, 0],
	[6640, 47326, 15563, 47289, 39408, 0, 0, 0],
	[7429, 47339, 15560, 47266, 40695, 4121, 0, 0],
	[8058, 47349, 15554, 47239, 41286, 0, 0, 0],
	[8387, 47357, 15545, 47209, 41915, 0, 0, 0],
	[8636, 47363, 15534, 47176, 42238, 4121, 0, 0],
	[8767, 47364, 15522, 47141, 42657, 4121, 0, 0],
	[9029, 47367, 15509, 47105, 43023, 4121, 0, 0],
	[9213, 47363, 15490, 47018, 43249, 4121, 0, 0],
	[9280, 47357, 15472, 46928, 43472, 5145, 0, 0],
	[9345, 47347, 15450, 46836, 43727, 5145, 0, 0],
	[9378, 47337, 15427, 46736, 43999, 5669, 0, 0],
	[9437, 47323, 15401, 46630, 44152, 5669, 0, 0],
	[9470, 47310, 15376, 46520, 44312, 6169, 0, 0],
	[9503, 47294, 15338, 46402, 44479, 6378, 0, 0],
	[9503, 47272, 15274, 46280, 44648, 6640, 0, 0],
	[9503, 47253, 15215, 46158, 44817, 6903, 0, 0],
	[9503, 47231, 15150, 45972, 45017, 7165, 0, 0],
	[9535, 47206, 15082, 45708, 45132, 7297, 0, 0],
	[9503, 47180, 15012, 45432, 45232, 7429, 0, 0],
	[9470, 47153, 14939, 45152, 45332, 7560, 0, 0],
	[9470, 47126, 14868, 44681, 45444, 7691, 0, 0],
	[9437, 47090, 14793, 44071, 45560, 7796, 0, 0],
	[9411, 47030, 14714, 42847, 45668, 7927, 0, 0],
	[9411, 46968, 14635, 8387, 45788, 8058, 0, 0],
	[9345, 46902, 14552, 10786, 45908, 8256, 0, 0],
	[9313, 46846, 14478, 11647, 46036, 8321, 0, 0],
	[9247, 46776, 14394, 12292, 46120, 8453, 0, 0],
	[9247, 46714, 14288, 12620, 46184, 8518, 0, 0],
	[9147, 46648, 14130, 12936, 46248, 8570, 0, 0],
	[9029, 46576, 13956, 13268, 46312, 8702, 0, 0],
	[8964, 46512, 13792, 13456, 46378, 8767, 0, 0],
	[8898, 46446, 13624, 13624, 46446, 8898, 0, 0],
	[8767, 46378, 13456, 13792, 46512, 8964, 0, 0],
	[8702, 46312, 13268, 13956, 46576, 9029, 0, 0],
	[8570, 46248, 12936, 14130, 46648, 9147, 0, 0],
	[8518, 46184, 12620, 14288, 46714, 9247, 0, 0],
	[8453, 46120, 12292, 14394, 46776, 9247, 0, 0],
	[8321, 46036, 11647, 14478, 46846, 9313, 0, 0],
	[8256, 45908, 10786, 14552, 46902, 9345, 0, 0],
	[8058, 45788, 8387, 14635, 46968, 9411, 0, 0],
	[7927, 45668, 42847, 14714, 47030, 9411, 0, 0],
	[7796, 45560, 44071, 14793, 47090, 9437, 0, 0],
	[7691, 45444, 44681, 14868, 47126, 9470, 0, 0],
	[7560, 45332, 45152, 14939, 47153, 9470, 0, 0],
	[7429, 45232, 45432, 15012, 47180, 9503, 0, 0],
	[7297, 45132, 45708, 15082, 47206, 9535, 0, 0],
	[7165, 45017, 45972, 15150, 47231, 9503, 0, 0],
	[6903, 44817, 46158, 15215, 47253, 9503, 0, 0],
	[6640, 44648, 46280, 15274, 47272, 9503, 0, 0],
	[6378, 44479, 46402, 15338, 47294, 9503, 0, 0],
	[6169, 44312, 46520, 15376, 47310, 9470, 0, 0],
	[5669, 44152, 46630, 15401, 47323, 9437, 0, 0],
	[5669, 43999, 46736, 15427, 47337, 9378, 0, 0],
	[5145, 43727, 46836, 15450, 47347, 9345, 0, 0],
	[5145, 43472, 46928, 15472, 47357, 9280, 0, 0],
	[4121, 43249, 47018, 15490, 47363, 9213, 0, 0],
	[4121, 43023, 47105, 15509, 47367, 9029, 0, 0],
	[4121, 42657, 47141, 15522, 47364, 8767, 0, 0],
	[4121, 42238, 47176, 15534, 47363, 8636, 0, 0],
	[0, 41915, 47209, 15545, 47357, 8387, 0, 0],
	[0, 41286, 47239, 15554, 47349, 8058, 0, 0],
	[4121, 40695, 47266, 15560, 47339, 7429, 0, 0],
	[0, 39408, 47289, 15563, 47326, 6640, 0, 0],
]

# coef_scale from NIS_Config.h (64x8, but only first 6 values used per phase)
coef_scale = [
	[0.0, 0.0, 1.0000, 0.0, 0.0, 0.0, 0.0, 0.0],
	[0.0029, -0.0127, 1.0000, 0.0132, -0.0034, 0.0, 0.0, 0.0],
	[0.0063, -0.0249, 0.9985, 0.0269, -0.0068, 0.0, 0.0, 0.0],
	[0.0088, -0.0361, 0.9956, 0.0415, -0.0103, 0.0005, 0.0, 0.0],
	[0.0117, -0.0474, 0.9932, 0.0562, -0.0142, 0.0005, 0.0, 0.0],
	[0.0142, -0.0576, 0.9897, 0.0713, -0.0181, 0.0005, 0.0, 0.0],
	[0.0166, -0.0674, 0.9844, 0.0874, -0.0220, 0.0010, 0.0, 0.0],
	[0.0186, -0.0762, 0.9785, 0.1040, -0.0264, 0.0015, 0.0, 0.0],
	[0.0205, -0.0850, 0.9727, 0.1206, -0.0308, 0.0020, 0.0, 0.0],
	[0.0225, -0.0928, 0.9648, 0.1382, -0.0352, 0.0024, 0.0, 0.0],
	[0.0239, -0.1006, 0.9575, 0.1558, -0.0396, 0.0029, 0.0, 0.0],
	[0.0254, -0.1074, 0.9487, 0.1738, -0.0439, 0.0034, 0.0, 0.0],
	[0.0264, -0.1138, 0.9390, 0.1929, -0.0488, 0.0044, 0.0, 0.0],
	[0.0278, -0.1191, 0.9282, 0.2119, -0.0537, 0.0049, 0.0, 0.0],
	[0.0288, -0.1245, 0.9170, 0.2310, -0.0581, 0.0059, 0.0, 0.0],
	[0.0293, -0.1294, 0.9058, 0.2510, -0.0630, 0.0063, 0.0, 0.0],
	[0.0303, -0.1333, 0.8926, 0.2710, -0.0679, 0.0073, 0.0, 0.0],
	[0.0308, -0.1367, 0.8789, 0.2915, -0.0728, 0.0083, 0.0, 0.0],
	[0.0308, -0.1401, 0.8657, 0.3120, -0.0776, 0.0093, 0.0, 0.0],
	[0.0313, -0.1426, 0.8506, 0.3330, -0.0825, 0.0103, 0.0, 0.0],
	[0.0313, -0.1445, 0.8354, 0.3540, -0.0874, 0.0112, 0.0, 0.0],
	[0.0313, -0.1460, 0.8193, 0.3755, -0.0923, 0.0122, 0.0, 0.0],
	[0.0313, -0.1470, 0.8022, 0.3965, -0.0967, 0.0137, 0.0, 0.0],
	[0.0308, -0.1479, 0.7856, 0.4185, -0.1016, 0.0146, 0.0, 0.0],
	[0.0303, -0.1479, 0.7681, 0.4399, -0.1060, 0.0156, 0.0, 0.0],
	[0.0298, -0.1479, 0.7505, 0.4614, -0.1104, 0.0166, 0.0, 0.0],
	[0.0293, -0.1470, 0.7314, 0.4829, -0.1147, 0.0181, 0.0, 0.0],
	[0.0288, -0.1460, 0.7119, 0.5049, -0.1187, 0.0190, 0.0, 0.0],
	[0.0278, -0.1445, 0.6929, 0.5264, -0.1226, 0.0200, 0.0, 0.0],
	[0.0273, -0.1431, 0.6724, 0.5479, -0.1260, 0.0215, 0.0, 0.0],
	[0.0264, -0.1411, 0.6528, 0.5693, -0.1299, 0.0225, 0.0, 0.0],
	[0.0254, -0.1387, 0.6323, 0.5903, -0.1328, 0.0234, 0.0, 0.0],
	[0.0244, -0.1357, 0.6113, 0.6113, -0.1357, 0.0244, 0.0, 0.0],
	[0.0234, -0.1328, 0.5903, 0.6323, -0.1387, 0.0254, 0.0, 0.0],
	[0.0225, -0.1299, 0.5693, 0.6528, -0.1411, 0.0264, 0.0, 0.0],
	[0.0215, -0.1260, 0.5479, 0.6724, -0.1431, 0.0273, 0.0, 0.0],
	[0.0200, -0.1226, 0.5264, 0.6929, -0.1445, 0.0278, 0.0, 0.0],
	[0.0190, -0.1187, 0.5049, 0.7119, -0.1460, 0.0288, 0.0, 0.0],
	[0.0181, -0.1147, 0.4829, 0.7314, -0.1470, 0.0293, 0.0, 0.0],
	[0.0166, -0.1104, 0.4614, 0.7505, -0.1479, 0.0298, 0.0, 0.0],
	[0.0156, -0.1060, 0.4399, 0.7681, -0.1479, 0.0303, 0.0, 0.0],
	[0.0146, -0.1016, 0.4185, 0.7856, -0.1479, 0.0308, 0.0, 0.0],
	[0.0137, -0.0967, 0.3965, 0.8022, -0.1470, 0.0313, 0.0, 0.0],
	[0.0122, -0.0923, 0.3755, 0.8193, -0.1460, 0.0313, 0.0, 0.0],
	[0.0112, -0.0874, 0.3540, 0.8354, -0.1445, 0.0313, 0.0, 0.0],
	[0.0103, -0.0825, 0.3330, 0.8506, -0.1426, 0.0313, 0.0, 0.0],
	[0.0093, -0.0776, 0.3120, 0.8657, -0.1401, 0.0308, 0.0, 0.0],
	[0.0083, -0.0728, 0.2915, 0.8789, -0.1367, 0.0308, 0.0, 0.0],
	[0.0073, -0.0679, 0.2710, 0.8926, -0.1333, 0.0303, 0.0, 0.0],
	[0.0063, -0.0630, 0.2510, 0.9058, -0.1294, 0.0293, 0.0, 0.0],
	[0.0059, -0.0581, 0.2310, 0.9170, -0.1245, 0.0288, 0.0, 0.0],
	[0.0049, -0.0537, 0.2119, 0.9282, -0.1191, 0.0278, 0.0, 0.0],
	[0.0044, -0.0488, 0.1929, 0.9390, -0.1138, 0.0264, 0.0, 0.0],
	[0.0034, -0.0439, 0.1738, 0.9487, -0.1074, 0.0254, 0.0, 0.0],
	[0.0029, -0.0396, 0.1558, 0.9575, -0.1006, 0.0239, 0.0, 0.0],
	[0.0024, -0.0352, 0.1382, 0.9648, -0.0928, 0.0225, 0.0, 0.0],
	[0.0020, -0.0308, 0.1206, 0.9727, -0.0850, 0.0205, 0.0, 0.0],
	[0.0015, -0.0264, 0.1040, 0.9785, -0.0762, 0.0186, 0.0, 0.0],
	[0.0010, -0.0220, 0.0874, 0.9844, -0.0674, 0.0166, 0.0, 0.0],
	[0.0005, -0.0181, 0.0713, 0.9897, -0.0576, 0.0142, 0.0, 0.0],
	[0.0005, -0.0142, 0.0562, 0.9932, -0.0474, 0.0117, 0.0, 0.0],
	[0.0005, -0.0103, 0.0415, 0.9956, -0.0361, 0.0088, 0.0, 0.0],
	[0.0, -0.0068, 0.0269, 0.9985, -0.0249, 0.0063, 0.0, 0.0],
	[0.0, -0.0034, 0.0132, 1.0000, -0.0127, 0.0029, 0.0, 0.0]
]

coef_usm = [
	[0.0, -0.6001, 1.2002, -0.6001, 0.0, 0.0, 0.0, 0.0],
	[0.0029, -0.6084, 1.1987, -0.5903, -0.0029, 0.0, 0.0, 0.0],
	[0.0049, -0.6147, 1.1958, -0.5791, -0.0068, 0.0005, 0.0, 0.0],
	[0.0073, -0.6196, 1.1890, -0.5659, -0.0103, 0.0, 0.0, 0.0],
	[0.0093, -0.6235, 1.1802, -0.5513, -0.0151, 0.0, 0.0, 0.0],
	[0.0112, -0.6265, 1.1699, -0.5352, -0.0195, 0.0005, 0.0, 0.0],
	[0.0122, -0.6270, 1.1582, -0.5181, -0.0259, 0.0005, 0.0, 0.0],
	[0.0142, -0.6284, 1.1455, -0.5005, -0.0317, 0.0005, 0.0, 0.0],
	[0.0156, -0.6265, 1.1274, -0.4790, -0.0386, 0.0005, 0.0, 0.0],
	[0.0166, -0.6235, 1.1089, -0.4570, -0.0454, 0.0010, 0.0, 0.0],
	[0.0176, -0.6187, 1.0879, -0.4346, -0.0532, 0.0010, 0.0, 0.0],
	[0.0181, -0.6138, 1.0659, -0.4102, -0.0615, 0.0015, 0.0, 0.0],
	[0.0190, -0.6069, 1.0405, -0.3843, -0.0698, 0.0015, 0.0, 0.0],
	[0.0195, -0.6006, 1.0161, -0.3574, -0.0796, 0.0020, 0.0, 0.0],
	[0.0200, -0.5928, 0.9893, -0.3286, -0.0898, 0.0024, 0.0, 0.0],
	[0.0200, -0.5820, 0.9580, -0.2988, -0.1001, 0.0029, 0.0, 0.0],
	[0.0200, -0.5728, 0.9292, -0.2690, -0.1104, 0.0034, 0.0, 0.0],
	[0.0200, -0.5620, 0.8975, -0.2368, -0.1226, 0.0039, 0.0, 0.0],
	[0.0205, -0.5498, 0.8643, -0.2046, -0.1343, 0.0044, 0.0, 0.0],
	[0.0200, -0.5371, 0.8301, -0.1709, -0.1465, 0.0049, 0.0, 0.0],
	[0.0195, -0.5239, 0.7944, -0.1367, -0.1587, 0.0054, 0.0, 0.0],
	[0.0195, -0.5107, 0.7598, -0.1021, -0.1724, 0.0059, 0.0, 0.0],
	[0.0190, -0.4966, 0.7231, -0.0649, -0.1865, 0.0063, 0.0, 0.0],
	[0.0186, -0.4819, 0.6846, -0.0288, -0.1997, 0.0068, 0.0, 0.0],
	[0.0186, -0.4668, 0.6460, 0.0093, -0.2144, 0.0073, 0.0, 0.0],
	[0.0176, -0.4507, 0.6055, 0.0479, -0.2290, 0.0083, 0.0, 0.0],
	[0.0171, -0.4370, 0.5693, 0.0859, -0.2446, 0.0088, 0.0, 0.0],
	[0.0161, -0.4199, 0.5283, 0.1255, -0.2598, 0.0098, 0.0, 0.0],
	[0.0161, -0.4048, 0.4883, 0.1655, -0.2754, 0.0103, 0.0, 0.0],
	[0.0151, -0.3887, 0.4497, 0.2041, -0.2910, 0.0107, 0.0, 0.0],
	[0.0142, -0.3711, 0.4072, 0.2446, -0.3066, 0.0117, 0.0, 0.0],
	[0.0137, -0.3555, 0.3672, 0.2852, -0.3228, 0.0122, 0.0, 0.0],
	[0.0132, -0.3394, 0.3262, 0.3262, -0.3394, 0.0132, 0.0, 0.0],
	[0.0122, -0.3228, 0.2852, 0.3672, -0.3555, 0.0137, 0.0, 0.0],
	[0.0117, -0.3066, 0.2446, 0.4072, -0.3711, 0.0142, 0.0, 0.0],
	[0.0107, -0.2910, 0.2041, 0.4497, -0.3887, 0.0151, 0.0, 0.0],
	[0.0103, -0.2754, 0.1655, 0.4883, -0.4048, 0.0161, 0.0, 0.0],
	[0.0098, -0.2598, 0.1255, 0.5283, -0.4199, 0.0161, 0.0, 0.0],
	[0.0088, -0.2446, 0.0859, 0.5693, -0.4370, 0.0171, 0.0, 0.0],
	[0.0083, -0.2290, 0.0479, 0.6055, -0.4507, 0.0176, 0.0, 0.0],
	[0.0073, -0.2144, 0.0093, 0.6460, -0.4668, 0.0186, 0.0, 0.0],
	[0.0068, -0.1997, -0.0288, 0.6846, -0.4819, 0.0186, 0.0, 0.0],
	[0.0063, -0.1865, -0.0649, 0.7231, -0.4966, 0.0190, 0.0, 0.0],
	[0.0059, -0.1724, -0.1021, 0.7598, -0.5107, 0.0195, 0.0, 0.0],
	[0.0054, -0.1587, -0.1367, 0.7944, -0.5239, 0.0195, 0.0, 0.0],
	[0.0049, -0.1465, -0.1709, 0.8301, -0.5371, 0.0200, 0.0, 0.0],
	[0.0044, -0.1343, -0.2046, 0.8643, -0.5498, 0.0205, 0.0, 0.0],
	[0.0039, -0.1226, -0.2368, 0.8975, -0.5620, 0.0200, 0.0, 0.0],
	[0.0034, -0.1104, -0.2690, 0.9292, -0.5728, 0.0200, 0.0, 0.0],
	[0.0029, -0.1001, -0.2988, 0.9580, -0.5820, 0.0200, 0.0, 0.0],
	[0.0024, -0.0898, -0.3286, 0.9893, -0.5928, 0.0200, 0.0, 0.0],
	[0.0020, -0.0796, -0.3574, 1.0161, -0.6006, 0.0195, 0.0, 0.0],
	[0.0015, -0.0698, -0.3843, 1.0405, -0.6069, 0.0190, 0.0, 0.0],
	[0.0015, -0.0615, -0.4102, 1.0659, -0.6138, 0.0181, 0.0, 0.0],
	[0.0010, -0.0532, -0.4346, 1.0879, -0.6187, 0.0176, 0.0, 0.0],
	[0.0010, -0.0454, -0.4570, 1.1089, -0.6235, 0.0166, 0.0, 0.0],
	[0.0005, -0.0386, -0.4790, 1.1274, -0.6265, 0.0156, 0.0, 0.0],
	[0.0005, -0.0317, -0.5005, 1.1455, -0.6284, 0.0142, 0.0, 0.0],
	[0.0005, -0.0259, -0.5181, 1.1582, -0.6270, 0.0122, 0.0, 0.0],
	[0.0005, -0.0195, -0.5352, 1.1699, -0.6265, 0.0112, 0.0, 0.0],
	[0.0, -0.0151, -0.5513, 1.1802, -0.6235, 0.0093, 0.0, 0.0],
	[0.0, -0.0103, -0.5659, 1.1890, -0.6196, 0.0073, 0.0, 0.0],
	[0.0005, -0.0068, -0.5791, 1.1958, -0.6147, 0.0049, 0.0, 0.0],
	[0.0, -0.0029, -0.5903, 1.1987, -0.6084, 0.0029, 0.0, 0.0]
]

def coef_fp16_to_hex_lines(coef):
	"""Convert coefficient array to hex string (64 rows, 2 rgba16hf per row)"""
	lines = []
	for row in coef:
		row_hex = ''
		for i in range(8):
			row_hex += struct.pack('<H', row[i]).hex()
		lines.append(row_hex)
	return '\n'.join(lines)

def coef_to_hex_lines(coef):
	"""Convert coefficient array to hex string (64 rows, 2 rgba32f per row)"""
	lines = []
	for row in coef:
		row_hex = ''
		# First vec4: row[0], row[1], row[2], row[3]
		for i in range(4):
			row_hex += struct.pack('<f', row[i]).hex()
		# Second vec4: row[4], row[5], 0, 0
		for i in range(4, 8):
			row_hex += struct.pack('<f', row[i]).hex()
		lines.append(row_hex)
	return '\n'.join(lines)

scale_hex_fp16 = coef_fp16_to_hex_lines(coef_scale_fp16)
scale_hex = coef_to_hex_lines(coef_scale)
usm_hex_fp16 = coef_fp16_to_hex_lines(coef_usm_fp16)
usm_hex = coef_to_hex_lines(coef_usm)

shader = '''// Ported from NVIDIA Image Scaling SDK


//!PARAM SHARP
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM NIS_HDR_MODE
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 2
0


//!HOOK MAIN
//!BIND HOOKED
//!BIND coef_scaler_fp16
//!BIND coef_usm_fp16
//!BIND coef_scaler
//!BIND coef_usm
//!DESC [NVScaler_RT] (SDK v1.0.3)
//!WIDTH OUTPUT.w
//!HEIGHT OUTPUT.h
//!WHEN OUTPUT.w HOOKED.w 1.0 * > OUTPUT.h HOOKED.h 1.0 * > *
//!COMPUTE 32 24 256 1

#define FP16   1

// Constants
#define kPhaseCount 64
#define kFilterSize 6
#define kSupportSize 6
#define kPadSize 6
#define NIS_BLOCK_WIDTH 32
#define NIS_BLOCK_HEIGHT 24
#define NIS_THREAD_GROUP_SIZE 256

#define kTilePitch (NIS_BLOCK_WIDTH + kPadSize)
#define kTileSize (kTilePitch * (NIS_BLOCK_HEIGHT + kPadSize))
#define kEdgeMapPitch (NIS_BLOCK_WIDTH + 2)
#define kEdgeMapSize (kEdgeMapPitch * (NIS_BLOCK_HEIGHT + 2))

#define kHDRCompressionFactor 0.282842712

shared float shPixelsY[kTileSize];
shared float shCoefScaler[kPhaseCount][kFilterSize];
shared float shCoefUSM[kPhaseCount][kFilterSize];
shared vec4 shEdgeMap[kEdgeMapSize];

// Sharpness (computed from SHARP)
float getDetectRatio() { return 2.0 * 1127.0 / 1024.0; }
float getDetectThres() {
#if NIS_HDR_MODE == 1 || NIS_HDR_MODE == 2
	return 32.0 / 1024.0;
#else
	return 64.0 / 1024.0;
#endif
}
float getMinContrastRatio() {
#if NIS_HDR_MODE == 1 || NIS_HDR_MODE == 2
	return 1.5;
#else
	return 2.0;
#endif
}
float getMaxContrastRatio() {
#if NIS_HDR_MODE == 1 || NIS_HDR_MODE == 2
	return 5.0;
#else
	return 10.0;
#endif
}

float getRatioNorm() {
	return 1.0 / (getMaxContrastRatio() - getMinContrastRatio());
}

float getSharpStartY() {
#if NIS_HDR_MODE == 2
	return 0.35;
#elif NIS_HDR_MODE == 1
	return 0.3;
#else
	return 0.45;
#endif
}

float getSharpEndY() {
#if NIS_HDR_MODE == 2
	return 0.55;
#elif NIS_HDR_MODE == 1
	return 0.5;
#else
	return 0.9;
#endif
}

float getSharpScaleY() {
	return 1.0 / (getSharpEndY() - getSharpStartY());
}

float getSharpStrengthMin() {
	float sharpen_slider = SHARP - 0.5;
	float MinScale = (sharpen_slider >= 0.0) ? 1.25 : 1.0;
#if NIS_HDR_MODE == 1 || NIS_HDR_MODE == 2
	return max(0.0, 0.4 + sharpen_slider * MinScale * 1.1);
#else
	return max(0.0, 0.4 + sharpen_slider * MinScale * 1.2);
#endif
}

float getSharpStrengthMax() {
	float sharpen_slider = SHARP - 0.5;
	float MaxScale = (sharpen_slider >= 0.0) ? 1.25 : 1.75;
#if NIS_HDR_MODE == 1 || NIS_HDR_MODE == 2
	return 2.2 + sharpen_slider * MaxScale * 1.8;
#else
	return 1.6 + sharpen_slider * MaxScale * 1.8;
#endif
}

float getSharpStrengthScale() {
	return getSharpStrengthMax() - getSharpStrengthMin();
}

float getSharpLimitMin() {
	float sharpen_slider = SHARP - 0.5;
	float LimitScale = (sharpen_slider >= 0.0) ? 1.25 : 1.0;
#if NIS_HDR_MODE == 1 || NIS_HDR_MODE == 2
	return max(0.06, 0.10 + sharpen_slider * LimitScale * 0.28);
#else
	return max(0.1, 0.14 + sharpen_slider * LimitScale * 0.32);
#endif
}

float getSharpLimitMax() {
	float sharpen_slider = SHARP - 0.5;
	float LimitScale = (sharpen_slider >= 0.0) ? 1.25 : 1.0;
#if NIS_HDR_MODE == 1 || NIS_HDR_MODE == 2
	return 0.6 + sharpen_slider * LimitScale * 0.6;
#else
	return 0.5 + sharpen_slider * LimitScale * 0.6;
#endif
}

float getSharpLimitScale() {
	return getSharpLimitMax() - getSharpLimitMin();
}

float getY(vec3 rgba) {
#if NIS_HDR_MODE == 2
	return 0.262 * rgba.r + 0.678 * rgba.g + 0.0593 * rgba.b;
#elif NIS_HDR_MODE == 1
	return sqrt(0.2126 * rgba.r + 0.7152 * rgba.g + 0.0722 * rgba.b) * kHDRCompressionFactor;
#else
	return 0.2126 * rgba.r + 0.7152 * rgba.g + 0.0722 * rgba.b;
#endif
}

float getYLinear(vec3 rgba) {
	return 0.2126 * rgba.r + 0.7152 * rgba.g + 0.0722 * rgba.b;
}

vec4 GetEdgeMap(float p[4][4], int i, int j) {
	float kDetectRatio = getDetectRatio();
	float kDetectThres = getDetectThres();

	float g_0 = abs(p[0 + i][0 + j] + p[0 + i][1 + j] + p[0 + i][2 + j] - p[2 + i][0 + j] - p[2 + i][1 + j] - p[2 + i][2 + j]);
	float g_45 = abs(p[1 + i][0 + j] + p[0 + i][0 + j] + p[0 + i][1 + j] - p[2 + i][1 + j] - p[2 + i][2 + j] - p[1 + i][2 + j]);
	float g_90 = abs(p[0 + i][0 + j] + p[1 + i][0 + j] + p[2 + i][0 + j] - p[0 + i][2 + j] - p[1 + i][2 + j] - p[2 + i][2 + j]);
	float g_135 = abs(p[1 + i][0 + j] + p[2 + i][0 + j] + p[2 + i][1 + j] - p[0 + i][1 + j] - p[0 + i][2 + j] - p[1 + i][2 + j]);

	float g_0_90_max = max(g_0, g_90);
	float g_0_90_min = min(g_0, g_90);
	float g_45_135_max = max(g_45, g_135);
	float g_45_135_min = min(g_45, g_135);

	if (g_0_90_max + g_45_135_max == 0.0) {
		return vec4(0.0);
	}

	float e_0_90 = min(g_0_90_max / (g_0_90_max + g_45_135_max), 1.0);
	float e_45_135 = 1.0 - e_0_90;

	bool c_0_90 = (g_0_90_max > (g_0_90_min * kDetectRatio)) && (g_0_90_max > kDetectThres) && (g_0_90_max > g_45_135_min);
	bool c_45_135 = (g_45_135_max > (g_45_135_min * kDetectRatio)) && (g_45_135_max > kDetectThres) && (g_45_135_max > g_0_90_min);
	bool c_g_0_90 = g_0_90_max == g_0;
	bool c_g_45_135 = g_45_135_max == g_45;

	float f_e_0_90 = (c_0_90 && c_45_135) ? e_0_90 : 1.0;
	float f_e_45_135 = (c_0_90 && c_45_135) ? e_45_135 : 1.0;

	float weight_0 = (c_0_90 && c_g_0_90) ? f_e_0_90 : 0.0;
	float weight_90 = (c_0_90 && !c_g_0_90) ? f_e_0_90 : 0.0;
	float weight_45 = (c_45_135 && c_g_45_135) ? f_e_45_135 : 0.0;
	float weight_135 = (c_45_135 && !c_g_45_135) ? f_e_45_135 : 0.0;

	return vec4(weight_0, weight_90, weight_45, weight_135);
}

void LoadFilterBanksSh(int i0) {
	int i = i0;
	if (i < kPhaseCount * 2) {
		int phase = i >> 1;
		int vIdx = i & 1;

#if FP16
		vec4 v = texelFetch(coef_scaler_fp16, ivec2(vIdx, phase), 0);
#else
		vec4 v = texelFetch(coef_scaler, ivec2(vIdx, phase), 0);
#endif
		int filterOffset = vIdx * 4;
		shCoefScaler[phase][filterOffset + 0] = v.x;
		shCoefScaler[phase][filterOffset + 1] = v.y;
		if (vIdx == 0) {
			shCoefScaler[phase][2] = v.z;
			shCoefScaler[phase][3] = v.w;
		}

#if FP16
		v = texelFetch(coef_usm_fp16, ivec2(vIdx, phase), 0);
#else
		v = texelFetch(coef_usm, ivec2(vIdx, phase), 0);
#endif
		shCoefUSM[phase][filterOffset + 0] = v.x;
		shCoefUSM[phase][filterOffset + 1] = v.y;
		if (vIdx == 0) {
			shCoefUSM[phase][2] = v.z;
			shCoefUSM[phase][3] = v.w;
		}
	}
}

float CalcLTI(float p0, float p1, float p2, float p3, float p4, float p5, int phase_index) {
	float kEps = 1.0 / 255.0;
	float kMinContrastRatio = getMinContrastRatio();
	float kRatioNorm = getRatioNorm();
	float kContrastBoost = 1.0;

	bool selector = (phase_index <= kPhaseCount / 2);
	float sel = selector ? p0 : p3;
	float a_min = min(min(p1, p2), sel);
	float a_max = max(max(p1, p2), sel);
	sel = selector ? p2 : p5;
	float b_min = min(min(p3, p4), sel);
	float b_max = max(max(p3, p4), sel);

	float a_cont = a_max - a_min;
	float b_cont = b_max - b_min;

	float cont_ratio = max(a_cont, b_cont) / (min(a_cont, b_cont) + kEps);
	return (1.0 - clamp((cont_ratio - kMinContrastRatio) * kRatioNorm, 0.0, 1.0)) * kContrastBoost;
}

vec4 GetInterpEdgeMap(vec4 edge[2][2], float phase_frac_x, float phase_frac_y) {
	vec4 h0 = mix(edge[0][0], edge[0][1], phase_frac_x);
	vec4 h1 = mix(edge[1][0], edge[1][1], phase_frac_x);
	return mix(h0, h1, phase_frac_y);
}

float EvalPoly6(float pxl[6], int phase_int) {
	float kSharpStartY = getSharpStartY();
	float kSharpScaleY = getSharpScaleY();
	float kSharpStrengthMin = getSharpStrengthMin();
	float kSharpStrengthScale = getSharpStrengthScale();
	float kSharpLimitMin = getSharpLimitMin();
	float kSharpLimitScale = getSharpLimitScale();

	float y = 0.0;
	for (int i = 0; i < 6; ++i) {
		y += shCoefScaler[phase_int][i] * pxl[i];
	}

	float y_usm = 0.0;
	for (int i = 0; i < 6; ++i) {
		y_usm += shCoefUSM[phase_int][i] * pxl[i];
	}

	float y_scale = 1.0 - clamp((y - kSharpStartY) * kSharpScaleY, 0.0, 1.0);
	float y_sharpness = y_scale * kSharpStrengthScale + kSharpStrengthMin;
	y_usm *= y_sharpness;

	float y_sharpness_limit = (y_scale * kSharpLimitScale + kSharpLimitMin) * y;
	y_usm = min(y_sharpness_limit, max(-y_sharpness_limit, y_usm));
	y_usm *= CalcLTI(pxl[0], pxl[1], pxl[2], pxl[3], pxl[4], pxl[5], phase_int);

	return y + y_usm;
}

float FilterNormal(float p[6][6], int phase_x_frac_int, int phase_y_frac_int) {
	float h_acc = 0.0;
	for (int j = 0; j < 6; ++j) {
		float v_acc = 0.0;
		for (int i = 0; i < 6; ++i) {
			v_acc += p[i][j] * shCoefScaler[phase_y_frac_int][i];
		}
		h_acc += v_acc * shCoefScaler[phase_x_frac_int][j];
	}
	return h_acc;
}

float AddDirFilters(float p[6][6], float phase_x_frac, float phase_y_frac, int phase_x_frac_int, int phase_y_frac_int, vec4 w) {
	float f = 0.0;

	if (w.x > 0.0) {
		float interp0Deg[6];
		for (int i = 0; i < 6; ++i) {
			interp0Deg[i] = mix(p[i][2], p[i][3], phase_x_frac);
		}
		f += EvalPoly6(interp0Deg, phase_y_frac_int) * w.x;
	}

	if (w.y > 0.0) {
		float interp90Deg[6];
		for (int i = 0; i < 6; ++i) {
			interp90Deg[i] = mix(p[2][i], p[3][i], phase_y_frac);
		}
		f += EvalPoly6(interp90Deg, phase_x_frac_int) * w.y;
	}

	if (w.z > 0.0) {
		float pphase_b45 = 0.5 + 0.5 * (phase_x_frac - phase_y_frac);

		float temp_interp45Deg[7];
		temp_interp45Deg[1] = mix(p[2][1], p[1][2], pphase_b45);
		temp_interp45Deg[3] = mix(p[3][2], p[2][3], pphase_b45);
		temp_interp45Deg[5] = mix(p[4][3], p[3][4], pphase_b45);

		float pb45 = pphase_b45 - 0.5;
		float a = (pb45 >= 0.0) ? p[0][2] : p[2][0];
		float b = (pb45 >= 0.0) ? p[1][3] : p[3][1];
		float c = (pb45 >= 0.0) ? p[2][4] : p[4][2];
		float d = (pb45 >= 0.0) ? p[3][5] : p[5][3];
		temp_interp45Deg[0] = mix(p[1][1], a, abs(pb45));
		temp_interp45Deg[2] = mix(p[2][2], b, abs(pb45));
		temp_interp45Deg[4] = mix(p[3][3], c, abs(pb45));
		temp_interp45Deg[6] = mix(p[4][4], d, abs(pb45));

		float interp45Deg[6];
		float pphase_p45 = phase_x_frac + phase_y_frac;
		if (pphase_p45 >= 1.0) {
			for (int i = 0; i < 6; i++) {
				interp45Deg[i] = temp_interp45Deg[i + 1];
			}
			pphase_p45 = pphase_p45 - 1.0;
		} else {
			for (int i = 0; i < 6; i++) {
				interp45Deg[i] = temp_interp45Deg[i];
			}
		}
		f += EvalPoly6(interp45Deg, int(pphase_p45 * 64.0)) * w.z;
	}

	if (w.w > 0.0) {
		float pphase_b135 = 0.5 * (phase_x_frac + phase_y_frac);

		float temp_interp135Deg[7];
		temp_interp135Deg[1] = mix(p[3][1], p[4][2], pphase_b135);
		temp_interp135Deg[3] = mix(p[2][2], p[3][3], pphase_b135);
		temp_interp135Deg[5] = mix(p[1][3], p[2][4], pphase_b135);

		float pb135 = pphase_b135 - 0.5;
		float a = (pb135 >= 0.0) ? p[5][2] : p[3][0];
		float b = (pb135 >= 0.0) ? p[4][3] : p[2][1];
		float c = (pb135 >= 0.0) ? p[3][4] : p[1][2];
		float d = (pb135 >= 0.0) ? p[2][5] : p[0][3];
		temp_interp135Deg[0] = mix(p[4][1], a, abs(pb135));
		temp_interp135Deg[2] = mix(p[3][2], b, abs(pb135));
		temp_interp135Deg[4] = mix(p[2][3], c, abs(pb135));
		temp_interp135Deg[6] = mix(p[1][4], d, abs(pb135));

		float interp135Deg[6];
		float pphase_p135 = 1.0 + (phase_x_frac - phase_y_frac);
		if (pphase_p135 >= 1.0) {
			for (int i = 0; i < 6; ++i) {
				interp135Deg[i] = temp_interp135Deg[i + 1];
			}
			pphase_p135 = pphase_p135 - 1.0;
		} else {
			for (int i = 0; i < 6; ++i) {
				interp135Deg[i] = temp_interp135Deg[i];
			}
		}
		f += EvalPoly6(interp135Deg, int(pphase_p135 * 64.0)) * w.w;
	}

	return f;
}

void hook() {

	ivec2 blockIdx = ivec2(gl_WorkGroupID.xy);
	uint threadIdx = gl_LocalInvocationIndex;

	float kScaleX = HOOKED_size.x / target_size.x;
	float kScaleY = HOOKED_size.y / target_size.y;

	int dstBlockX = NIS_BLOCK_WIDTH * blockIdx.x;
	int dstBlockY = NIS_BLOCK_HEIGHT * blockIdx.y;

	// Calculate source block bounds
	int srcBlockStartX = int(floor((float(dstBlockX) + 0.5) * kScaleX - 0.5));
	int srcBlockStartY = int(floor((float(dstBlockY) + 0.5) * kScaleY - 0.5));
	int srcBlockEndX = int(ceil((float(dstBlockX + NIS_BLOCK_WIDTH) + 0.5) * kScaleX - 0.5));
	int srcBlockEndY = int(ceil((float(dstBlockY + NIS_BLOCK_HEIGHT) + 0.5) * kScaleY - 0.5));

	int numTilePixelsX = srcBlockEndX - srcBlockStartX + kSupportSize - 1;
	int numTilePixelsY = srcBlockEndY - srcBlockStartY + kSupportSize - 1;

	numTilePixelsX += numTilePixelsX & 1;
	numTilePixelsY += numTilePixelsY & 1;
	int numTilePixels = numTilePixelsX * numTilePixelsY;

	int numEdgeMapPixelsX = numTilePixelsX - kSupportSize + 2;
	int numEdgeMapPixelsY = numTilePixelsY - kSupportSize + 2;
	int numEdgeMapPixels = numEdgeMapPixelsX * numEdgeMapPixelsY;

	// Load luma tile into shared memory
	// Shift by -2.0 to center the 6-tap filter support (filter taps at -2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
	for (uint i = threadIdx * 2u; i < uint(numTilePixels) >> 1; i += NIS_THREAD_GROUP_SIZE * 2u) {
		uint py = (i / uint(numTilePixelsX)) * 2u;
		uint px = i % uint(numTilePixelsX);

		float kShift = -2.0;  // Center of 6-tap filter
		float srcX = float(srcBlockStartX) + float(px) + kShift;
		float srcY = float(srcBlockStartY) + float(py) + kShift;

		float p[2][2];
#ifdef HOOKED_gather
		{
			float ksrcX = (float(srcBlockStartX) + float(px) + kShift + 0.5) * HOOKED_pt.x;
			float ksrcY = (float(srcBlockStartY) + float(py) + kShift + 0.5) * HOOKED_pt.y;
			vec4 sr = HOOKED_gather(vec2(ksrcX, ksrcY), 0);
			vec4 sg = HOOKED_gather(vec2(ksrcX, ksrcY), 1);
			vec4 sb = HOOKED_gather(vec2(ksrcX, ksrcY), 2);

			p[0][0] = getY(vec3(sr.w, sg.w, sb.w));
			p[0][1] = getY(vec3(sr.z, sg.z, sb.z));
			p[1][0] = getY(vec3(sr.x, sg.x, sb.x));
			p[1][1] = getY(vec3(sr.y, sg.y, sb.y));
		}
#else
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				// Convert to normalized texture coordinates with 0.5 texel center offset
				float tx = (srcX + float(k) + 0.5) * HOOKED_pt.x;
				float ty = (srcY + float(j) + 0.5) * HOOKED_pt.y;
				vec4 px_color = HOOKED_tex(vec2(tx, ty));
				p[j][k] = getY(px_color.rgb);
			}
		}
#endif
		uint idx = py * uint(kTilePitch) + px;
		shPixelsY[idx] = p[0][0];
		shPixelsY[idx + 1u] = p[0][1];
		shPixelsY[idx + uint(kTilePitch)] = p[1][0];
		shPixelsY[idx + uint(kTilePitch) + 1u] = p[1][1];
	}

	barrier();

	// Compute edge map
	for (uint i = threadIdx * 2u; i < uint(numEdgeMapPixels) >> 1; i += NIS_THREAD_GROUP_SIZE * 2u) {
		uint py = (i / uint(numEdgeMapPixelsX)) * 2u;
		uint px = i % uint(numEdgeMapPixelsX);

		uint edgeMapIdx = py * uint(kEdgeMapPitch) + px;
		uint tileCornerIdx = (py + 1u) * uint(kTilePitch) + px + 1u;

		float p[4][4];
		for (int j = 0; j < 4; j++) {
			for (int k = 0; k < 4; k++) {
				p[j][k] = shPixelsY[tileCornerIdx + uint(j) * uint(kTilePitch) + uint(k)];
			}
		}

		shEdgeMap[edgeMapIdx] = GetEdgeMap(p, 0, 0);
		shEdgeMap[edgeMapIdx + 1u] = GetEdgeMap(p, 0, 1);
		shEdgeMap[edgeMapIdx + uint(kEdgeMapPitch)] = GetEdgeMap(p, 1, 0);
		shEdgeMap[edgeMapIdx + uint(kEdgeMapPitch) + 1u] = GetEdgeMap(p, 1, 1);
	}

	LoadFilterBanksSh(int(threadIdx));
	barrier();

	ivec2 pos = ivec2(uint(threadIdx) % uint(NIS_BLOCK_WIDTH), uint(threadIdx) / uint(NIS_BLOCK_WIDTH));
	int dstX = dstBlockX + pos.x;
	float srcX = (0.5 + float(dstX)) * kScaleX - 0.5;
	int px_pos = int(floor(srcX) - float(srcBlockStartX));
	float fx = srcX - floor(srcX);
	int fx_int = int(fx * float(kPhaseCount));

	for (int k = 0; k < NIS_BLOCK_WIDTH * NIS_BLOCK_HEIGHT / NIS_THREAD_GROUP_SIZE; ++k) {
		int dstY = dstBlockY + pos.y + k * (NIS_THREAD_GROUP_SIZE / NIS_BLOCK_WIDTH);
		float srcY = (0.5 + float(dstY)) * kScaleY - 0.5;

		int py_pos = int(floor(srcY) - float(srcBlockStartY));
		float fy = srcY - floor(srcY);
		int fy_int = int(fy * float(kPhaseCount));

		int startEdgeMapIdx = py_pos * kEdgeMapPitch + px_pos;
		vec4 edge[2][2];
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				edge[i][j] = shEdgeMap[startEdgeMapIdx + i * kEdgeMapPitch + j];
			}
		}
		vec4 w = GetInterpEdgeMap(edge, fx, fy);

		int startTileIdx = py_pos * kTilePitch + px_pos;
		float p[6][6];
		for (int i = 0; i < 6; ++i) {
			for (int j = 0; j < 6; ++j) {
				p[i][j] = shPixelsY[startTileIdx + i * kTilePitch + j];
			}
		}

		float baseWeight = 1.0 - w.x - w.y - w.z - w.w;

		float opY = 0.0;
		opY += FilterNormal(p, fx_int, fy_int) * baseWeight;
		opY += AddDirFilters(p, fx, fy, fx_int, fy_int, w);

		// Sample the source at the corresponding position
		float srcTexX = (srcX + 0.5) * HOOKED_pt.x;
		float srcTexY = (srcY + 0.5) * HOOKED_pt.y;
		ivec2 dstCoord = ivec2(dstX, dstY);

		if (dstX >= int(target_size.x) || dstY >= int(target_size.y)) {
			continue;
		}

		vec4 op = HOOKED_tex(vec2(srcTexX, srcTexY));
		float y = getY(op.rgb);

#if NIS_HDR_MODE == 1
		float kEps = 1e-4;
		float opYN = max(opY, 0.0) / kHDRCompressionFactor;
		float corr = (opYN * opYN + kEps) / (max(getYLinear(op.rgb), 0.0) + kEps);
		op.rgb *= corr;
#else
		float corr = opY - y;
		op.rgb += corr;
#endif

		op = clamp(op, 0.0, 1.0);
		imageStore(out_image, dstCoord, op);
	}

}

//!TEXTURE coef_scaler_fp16
//!SIZE 2 64
//!FORMAT rgba16hf
//!FILTER NEAREST
''' + scale_hex_fp16 + '''

//!TEXTURE coef_usm_fp16
//!SIZE 2 64
//!FORMAT rgba16hf
//!FILTER NEAREST
''' + usm_hex_fp16 + '''

//!TEXTURE coef_scaler
//!SIZE 2 64
//!FORMAT rgba32f
//!FILTER NEAREST
''' + scale_hex + '''

//!TEXTURE coef_usm
//!SIZE 2 64
//!FORMAT rgba32f
//!FILTER NEAREST
''' + usm_hex + '''

'''

with open('NVScaler_RT.glsl', 'w', encoding='utf-8', newline='\n') as f:
	f.write(shader)

print("Generated NVScaler_RT.glsl")
