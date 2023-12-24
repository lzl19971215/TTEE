# -*- coding: utf-8 -*-
NER_LABEL_MAPPING = {
    'O': 0,
    'B': 1,
    'I': 2,
    'E': 3,
    'S': 4
}

NER_BIO_MAPPING = {
    'B': 0,
    'I': 1,
    'O': 2
}

CATEGORY_LABEL_MAPPING = {
    'AMBIENCE#GENERAL': 0,
    'DRINKS#PRICES': 1,
    'DRINKS#QUALITY': 2,
    'DRINKS#STYLE_OPTIONS': 3,
    'FOOD#PRICES': 4,
    'FOOD#QUALITY': 5,
    'FOOD#STYLE_OPTIONS': 6,
    'LOCATION#GENERAL': 7,
    'RESTAURANT#GENERAL': 8,
    'RESTAURANT#MISCELLANEOUS': 9,
    'RESTAURANT#PRICES': 10,
    'SERVICE#GENERAL': 11
}

LABEL_CATEGORY_MAPPING = {v: k for k, v in CATEGORY_LABEL_MAPPING.items()}

SENTIMENT_LABEL_MAPPING = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}

LABEL_SENTIMENT_MAPPING = {v: k for k, v in SENTIMENT_LABEL_MAPPING.items()}

SENTENCE_B = {
    "cased": {
        "text": "general of ambience price of drinks quality of drinks style options of drinks prices of food "
                "quality of food style options of food general of location general of restaurant "
                "miscellaneous of restaurant prices of restaurant general of service",
        "aspect_term_mapping": {
            0: (0, 5),
            1: (5, 8),
            2: (8, 11),
            3: (11, 15),
            4: (15, 18),
            5: (18, 21),
            6: (21, 25),
            7: (25, 28),
            8: (28, 31),
            9: (31, 37),
            10: (37, 40),
            11: (40, 43)
        }
    },
    "uncased": {
        "text": "general of ambience price of drinks quality of drinks style options of drinks prices of food "
                "quality of food style options of food general of location general of restaurant "
                "miscellaneous of restaurant prices of restaurant general of service",
        "aspect_term_mapping": {
            0: (0, 5),
            1: (5, 8),
            2: (8, 11),
            3: (11, 15),
            4: (15, 18),
            5: (18, 21),
            6: (21, 25),
            7: (25, 28),
            8: (28, 31),
            9: (31, 34),
            10: (34, 37),
            11: (37, 40)
        }
    }

}

# "texts": [
#     "ambience general",
#     "drinks price",
#     "drinks quality",
#     "drinks style options",
#     "food prices",
#     "food quality",
#     "food style options",
#     "location general",
#     "restaurant general",
#     "restaurant miscellaneous",
#     "restaurant prices",
#     "service general"
# ],
ASPECT_SENTENCE = {
    "texts": [
        "general of ambience",
        "price of drinks",
        "quality of drinks",
        "style options of drinks",
        "prices of food",
        "quality of food",
        "style options of food",
        "general of location",
        "general of restaurant",
        "miscellaneous of restaurant",
        "prices of restaurant",
        "general of service"
    ],

    "sentiments": [
        "negative",
        "neutral",
        "positive"

    ]
}

ASPECT_SENTENCE_ACOS_LAPTOP = {
    "texts": [
        "general of ambience",
        "price of drinks",
        "quality of drinks",
        "style options of drinks",
        "prices of food",
        "quality of food",
        "style options of food",
        "general of location",
        "general of restaurant",
        "miscellaneous of restaurant",
        "prices of restaurant",
        "general of service"
    ],

    "sentiments": [
        "negative",
        "neutral",
        "positive"
    ]
}

NUM_CATEGORIES = 12


# CHINESE

ASPECT_SENTENCE_CHINESE = {
    "texts": [
        "显示：屏幕、显示",
        "电源：电池、电源、充电、续航、待机",
        "系统性能：操作系统、处理器、内存、性能",
        "硬件设备：硬件、设备、存储、零件、容量",
        "软件应用：软件、程序、游戏、应用",
        "键盘输入：键盘、按键、按钮、输入",
        "移动通信：网络、信号、传输、接口、通话、蓝牙",
        "多媒体：多媒体、视频、音乐、拍照、摄影、音响",
        "服务支持：销售、售后、服务支持",
        "价格：售价、价格",
        "综合：手机综合、整体"
    ],
    "sentiments": [
        "负面",
        "中性",
        "正面"
    ]
}

CATEGORY_LABEL_MAPPING_CHINESE = {
    '显示': 0,
    '电源': 1,
    '系统性能': 2,
    '硬件设备': 3,
    '软件应用': 4,
    '键盘输入': 5,
    '移动通信': 6,
    '多媒体': 7,
    '服务支持': 8,
    '价格': 9,
    '综合': 10
}

SENTIMENT_LABEL_MAPPING_CHINESE = {
    '负面': 0,
    '中性': 1,
    '正面': 2
}

LABEL_CATEGORY_MAPPING_CHINESE = {v: k for k, v in CATEGORY_LABEL_MAPPING_CHINESE.items()}

LABEL_SENTIMENT_MAPPING_CHINESE = {v: k for k, v in SENTIMENT_LABEL_MAPPING_CHINESE.items()}




