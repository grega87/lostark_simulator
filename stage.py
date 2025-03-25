from enum import Enum
from copy import deepcopy
import random

import torch

CardProbabilities = [0.15, 0.115, 0.095, 0.055, 0.105, 0.07, 0.07, 0.09, 0.1, 0.15]
PercentByDistance = [100, 85, 70, 55, 40, 25, 10, 10]
ProbByDistance = [1, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1]
BlockProbabilities = [0.235, 0.115, 0.16, 0.16, 0.16, 0.17]


class CardDefine(Enum):
    NONE = 0
    # 위, 아래, 좌, 우 1칸 범위 확률
    # 1레벨 확률 50%
    LIGHTNING = 1       # 낙뢰
    FIRE = 2            # 업화
    EARTH = 3           # 충격파
    WATER = 4           # 해일
    EXPLOSION = 5       # 대폭발
    EARTHQUAKE = 6      # 지진
    TYPHOON = 7         # 폭풍우
    THUNDER = 8         # 벼락
    PURIFICATION = 9    # 정화
    WATERSPOUT = 10     # 용오름
    ERUPTION = 11       # 분출
    WORLD_TREE = 12     # 세계수의 공명
    SPECIAL_CARD = [PURIFICATION, WORLD_TREE]


# 19 * 19 공간 중심에서 카드를 쓸 때 블록을 부술 확률
class CardProbabilityROI:
    MAX_WIDTH_HEIGHT = 15       # 가로 세로 맥스 값을 10 x 10 이라고 가정
    CENTER_POS = 7
    TOTAL_CARD_NUM = 13
    MAX_CARD_LEVEL = 4

    def __init__(self, device=torch.device('cpu')):
        self.device = device

        # 0번째 카드 정보, 0번째 카드 정보는 의미가 없지만 인덱싱 편의를 위해 생성
        self.base_prob_map = torch.zeros(
            (self.TOTAL_CARD_NUM, self.MAX_CARD_LEVEL, self.MAX_WIDTH_HEIGHT, self.MAX_WIDTH_HEIGHT),
            dtype=torch.float32,
            device=device
        )

        x_pos = torch.tensor([0, 0, -1, 1], dtype=torch.long, device=device) + self.CENTER_POS
        y_pos = torch.tensor([-1, 1, 0, 0], dtype=torch.long, device=device) + self.CENTER_POS
        card_type = torch.tensor([CardDefine.LIGHTNING.value] * len(x_pos), dtype=torch.long, device=device)
        self.base_prob_map[card_type, 1, y_pos, x_pos] = 0.5
        self.base_prob_map[card_type, 2:, y_pos, x_pos] = 1.

        x_pos = torch.tensor([0, 0, 0, 0, 1, 1, 1, -1, -1, -1, 2, -2, 0, 0], dtype=torch.long, device=device) + self.CENTER_POS
        y_pos = torch.tensor([-2, -1, 1, 2, -1, 0, 1, -1, 0, 1, 0, 0, 2, -2], dtype=torch.long, device=device) + self.CENTER_POS
        card_type = torch.tensor([CardDefine.FIRE.value] * len(x_pos), dtype=torch.long, device=device)
        self.base_prob_map[card_type, 1, y_pos, x_pos] = 0.5
        self.base_prob_map[card_type, 2:, y_pos, x_pos] = 1.

        x_pos = torch.tensor([-1, -1, -1, 0, 0, 1, 1, 1], dtype=torch.long, device=device) + self.CENTER_POS
        y_pos = torch.tensor([-1, 0, 1, -1, 1, -1, 0, 1], dtype=torch.long, device=device) + self.CENTER_POS
        card_type = torch.tensor([CardDefine.EARTH.value] * len(x_pos), dtype=torch.long, device=device)
        self.base_prob_map[card_type, 1, y_pos, x_pos] = 0.75
        self.base_prob_map[card_type, 2:, y_pos, x_pos] = 1.

        prob_map = torch.tensor(ProbByDistance[1:], dtype=torch.float32, device=device)
        move_step = torch.arange(1, self.CENTER_POS + 1, dtype=torch.long, device=device)
        move_positive_from_center = self.CENTER_POS + move_step
        move_negative_from_center = self.CENTER_POS - move_step

        card_type = torch.tensor([CardDefine.WATER.value] * len(prob_map), dtype=torch.long, device=device)
        self.base_prob_map[card_type, 1, move_positive_from_center, self.CENTER_POS] = prob_map
        self.base_prob_map[card_type, 1, move_negative_from_center, self.CENTER_POS] = prob_map
        self.base_prob_map[card_type, 1, self.CENTER_POS, move_positive_from_center] = prob_map
        self.base_prob_map[card_type, 1, self.CENTER_POS, move_negative_from_center] = prob_map
        self.base_prob_map[card_type, 2:, move_positive_from_center, self.CENTER_POS] = 1.
        self.base_prob_map[card_type, 2:, move_negative_from_center, self.CENTER_POS] = 1.
        self.base_prob_map[card_type, 2:, self.CENTER_POS, move_positive_from_center] = 1.
        self.base_prob_map[card_type, 2:, self.CENTER_POS, move_negative_from_center] = 1.

        card_type = torch.tensor([CardDefine.EXPLOSION.value] * len(prob_map), dtype=torch.long, device=device)
        self.base_prob_map[card_type, 1, move_positive_from_center, move_positive_from_center] = prob_map
        self.base_prob_map[card_type, 1, move_negative_from_center, move_negative_from_center] = prob_map
        self.base_prob_map[card_type, 1, move_positive_from_center, move_negative_from_center] = prob_map
        self.base_prob_map[card_type, 1, move_negative_from_center, move_positive_from_center] = prob_map
        self.base_prob_map[card_type, 2:, move_positive_from_center, move_positive_from_center] = 1.
        self.base_prob_map[card_type, 2:, move_negative_from_center, move_negative_from_center] = 1.
        self.base_prob_map[card_type, 2:, move_positive_from_center, move_negative_from_center] = 1.
        self.base_prob_map[card_type, 2:, move_negative_from_center, move_positive_from_center] = 1.

        card_type = torch.tensor([CardDefine.EARTHQUAKE.value] * len(prob_map), dtype=torch.long, device=device)
        self.base_prob_map[card_type, 1, self.CENTER_POS, move_positive_from_center] = prob_map
        self.base_prob_map[card_type, 1, self.CENTER_POS, move_negative_from_center] = prob_map
        self.base_prob_map[card_type, 2, self.CENTER_POS, move_positive_from_center] = 1.
        self.base_prob_map[card_type, 2, self.CENTER_POS, move_negative_from_center] = 1.
        self.base_prob_map[card_type, 3, self.CENTER_POS, move_positive_from_center] = 1.
        self.base_prob_map[card_type, 3, self.CENTER_POS, move_negative_from_center] = 1.

        card_type = torch.tensor([CardDefine.TYPHOON.value] * len(prob_map), dtype=torch.long, device=device)
        self.base_prob_map[card_type, 1, move_positive_from_center, self.CENTER_POS] = prob_map
        self.base_prob_map[card_type, 1, move_negative_from_center, self.CENTER_POS] = prob_map
        self.base_prob_map[card_type, 2, move_positive_from_center, self.CENTER_POS] = 1.
        self.base_prob_map[card_type, 2, move_negative_from_center, self.CENTER_POS] = 1.
        self.base_prob_map[card_type, 3, move_positive_from_center, self.CENTER_POS] = 1.
        self.base_prob_map[card_type, 3, move_negative_from_center, self.CENTER_POS] = 1.

        # 번개의 경우, 확률 값을 전달하지 않음
        # card_type = torch.tensor([CardDefine.THUNDER.value] * len(prob_map), dtype=torch.long, device=device)
        # self.base_prob_map[card_type, 1, :, :] = 0.1
        # self.base_prob_map[card_type, 2, :, :] = 0.2
        # self.base_prob_map[card_type, 3, :, :] = 0.3

        x_pos = torch.tensor([0, 0, 1, -1], dtype=torch.long, device=device) + self.CENTER_POS
        y_pos = torch.tensor([-1, 1, 0, 0], dtype=torch.long, device=device) + self.CENTER_POS
        card_type = torch.tensor([CardDefine.PURIFICATION.value] * len(x_pos), dtype=torch.long, device=device)

        self.base_prob_map[card_type[2:], 1, y_pos[2:], x_pos[2:]] = 0.5
        self.base_prob_map[card_type[2:], 2, y_pos[2:], x_pos[2:]] = 1.
        self.base_prob_map[card_type, 3, y_pos, x_pos] = 1.

        x_pos = torch.tensor([-1, -1, 1, 1], dtype=torch.long, device=device) + self.CENTER_POS
        y_pos = torch.tensor([-1, 1, -1, 1], dtype=torch.long, device=device) + self.CENTER_POS
        card_type = torch.tensor([CardDefine.WATERSPOUT.value] * len(x_pos), dtype=torch.long, device=device)

        self.base_prob_map[card_type, 1, y_pos, x_pos] = 0.5
        self.base_prob_map[card_type, 2:, y_pos, x_pos] = 1.

        # self.Eruption = deepcopy(self.base_prob_map)

        x_pos = torch.tensor([0, 0, 0, 0, -2, -1, 1, 2], dtype=torch.long, device=device) + self.CENTER_POS
        y_pos = torch.tensor([-2, -1, 1, 2, 0, 0, 0, 0], dtype=torch.long, device=device) + self.CENTER_POS
        card_type = torch.tensor([CardDefine.WORLD_TREE.value] * len(x_pos), dtype=torch.long, device=device)

        # 세계수의 공명은 모든 카드 레벨에 동일 값 설정
        self.base_prob_map[card_type, :, y_pos, x_pos] = 1.

        # 정 중앙 확률은 모두 1로 지정
        self.base_prob_map[:, :, self.CENTER_POS, self.CENTER_POS] = 1.

    def get_card_roi(self, card_id, card_level):
        return self.base_prob_map[card_id, card_level]

    def verify_device(self, device):
        if self.device != device:
            if device == torch.device('cpu'):
                set_device = device
            else:
                set_device = torch.device('cuda')
            self.base_prob_map = self.base_prob_map.to(set_device)
            self.device = set_device



class ActionDefine(Enum):
    NO_ACTION = 0
    USE_CARD_1 = 1
    USE_CARD_2 = 2
    CHANGE_CARD_1 = 3
    CHANGE_CARD_2 = 4


class BlockDefine(Enum):
    EMPTY = 0 #비어있음
    DESTROYABLE = 1 # 파괴 가능한 블록
    DESTROYED = 2 # 파괴된 블록
    DISTORTION = 3 # 왜곡된 블록
    ADDITION = 4 # 카드 교체 횟수 추가
    BLESS = 5 # 축복 (턴 - 1)
    MYSTIC = 6 # 세계수의 나무 or 분출
    ENFORCE = 7 # 다른 카드 강화
    COPY = 8 # 카드 복사
    RELOCATION = 9 # 블록 재배치
    BREAKABLE = [DESTROYABLE, ADDITION, BLESS, MYSTIC, ENFORCE, COPY, RELOCATION]


BlockList = [BlockDefine.ADDITION.value, BlockDefine.BLESS.value, BlockDefine.MYSTIC.value, BlockDefine.ENFORCE.value,
             BlockDefine.COPY.value, BlockDefine.RELOCATION.value]


class Stage:
    head1 = [[0, 1, 1, 1, 1, 0],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 0]]

    head2 = [[0, 1, 1, 1, 1, 0],
              [1, 3, 1, 1, 1, 1],
              [1, 1, 3, 1, 1, 1],
              [1, 1, 1, 3, 1, 1],
              [1, 1, 1, 1, 3, 1],
              [0, 1, 1, 1, 1, 0]]

    head3 = [[0, 1, 1, 1, 1, 0],
              [1, 3, 1, 1, 1, 1],
              [1, 1, 3, 1, 1, 1],
              [1, 1, 1, 3, 1, 1],
              [1, 1, 1, 1, 3, 1],
              [0, 1, 1, 1, 1, 0]]

    head4 = [[0, 0, 1, 1, 1, 0, 0],
              [0, 1, 3, 1, 1, 1, 0],
              [1, 1, 1, 1, 1, 3, 1],
              [1, 1, 1, 3, 1, 1, 1],
              [1, 3, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 3, 1, 0],
             [0, 0, 1, 1, 1, 0, 0]]

    head5 = [[0, 0, 1, 1, 1, 0, 0],
              [0, 1, 3, 1, 3, 1, 0],
              [1, 1, 1, 1, 1, 1, 1],
              [1, 3, 1, 1, 1, 3, 1],
              [1, 1, 1, 1, 1, 1, 1],
              [0, 1, 3, 1, 3, 1, 0],
             [0, 0, 1, 1, 1, 0, 0]]

    head6 = [[0, 0, 1, 1, 1, 1, 0, 0],
              [0, 1, 3, 1, 1, 1, 1, 0],
              [1, 1, 1, 3, 1, 1, 3, 1],
              [1, 1, 1, 1, 1, 3, 1, 1],
              [1, 1, 3, 1, 1, 1, 1, 1],
              [1, 3, 1, 1, 3, 1, 1, 1],
              [0, 1, 1, 1, 1, 3, 1, 0],
              [0, 0, 1, 1, 1, 1, 0, 0]]

    head7 = [[0, 0, 1, 1, 1, 1, 0, 0],
              [0, 1, 3, 3, 1, 1, 1, 0],
              [1, 1, 1, 1, 1, 1, 3, 1],
              [1, 1, 1, 3, 1, 1, 3, 1],
              [1, 3, 1, 1, 3, 1, 1, 1],
              [1, 3, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 3, 3, 1, 0],
              [0, 0, 1, 1, 1, 1, 0, 0]]

    shoulder1 = [[1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1,],
              [1, 1, 1, 1, 1, 1,],
              [1, 1, 1, 1, 1, 1,],
              [1, 1, 1, 1, 1, 1,]]

    shoulder2 = [[1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 3, 1],
              [1, 1, 1, 3, 1, 1,],
              [1, 1, 3, 1, 1, 1,],
              [1, 3, 1, 1, 1, 1,],
              [1, 1, 1, 1, 1, 1,]]

    shoulder3 = [[1, 1, 1, 1, 1, 1],
              [1, 3, 1, 1, 3, 1],
              [1, 1, 1, 1, 1, 1,],
              [1, 1, 1, 1, 1, 1,],
              [1, 3, 1, 1, 3, 1,],
              [1, 1, 1, 1, 1, 1,]]

    shoulder4 = [[1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 3, 1, 1, 1],
              [1, 3, 1, 1, 1, 3, 1],
              [1, 1, 1, 1, 1, 1, 1],
              [1, 3, 1, 1, 1, 3, 1],
              [1, 1, 1, 3, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1]]

    shoulder5 = [[1, 1, 1, 1, 1, 1, 1],
              [1, 3, 1, 1, 3, 1, 1],
              [1, 1, 1, 1, 1, 3, 1],
              [1, 1, 1, 3, 1, 1, 1],
              [1, 3, 1, 1, 1, 1, 1],
              [1, 1, 3, 1, 1, 3, 1],
             [1, 1, 1, 1, 1, 1, 1]]

    shoulder6 = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 3, 1, 1],
              [1, 3, 1, 3, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 3, 1, 1],
              [1, 1, 3, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 3, 1, 3, 1],
              [1, 1, 3, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

    shoulder7 = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 3, 1, 1, 3, 1, 1],
              [1, 3, 1, 1, 1, 1, 3, 1],
              [1, 1, 1, 3, 1, 1, 1, 1],
              [1, 1, 1, 1, 3, 1, 1, 1],
              [1, 3, 1, 1, 1, 1, 3, 1],
              [1, 1, 3, 1, 1, 3, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

    body1 =[[0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1,],
            [1, 1, 1, 1, 1, 1,],
            [0, 1, 1, 1, 1, 0,],
            [0, 0, 1, 1, 0, 0,]]

    body2 =[[0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 3, 3, 1, 1,],
            [1, 1, 3, 3, 1, 1,],
            [0, 1, 1, 1, 1, 0,],
            [0, 0, 1, 1, 0, 0,]]

    body3 =[[0, 0, 1, 1, 0, 0],
            [0, 1, 3, 1, 1, 0],
            [1, 1, 1, 1, 3, 1,],
            [1, 3, 1, 1, 1, 1,],
            [0, 1, 1, 3, 1, 0,],
            [0, 0, 1, 1, 0, 0,]]

    body4 =  [[0, 0, 0, 1, 0, 0, 0],
              [0, 0, 1, 1, 3, 0, 0],
              [0, 3, 3, 1, 1, 1, 0],
              [1, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 3, 3, 0],
              [0, 0, 3, 1, 1, 0, 0],
              [0, 0, 0, 1, 0, 0, 0]]

    body5 = [[0, 0, 0, 1, 0, 0, 0],
              [0, 0, 3, 1, 1, 0, 0],
              [0, 1, 1, 1, 1, 3, 0],
              [1, 1, 1, 3, 1, 1, 1],
              [0, 3, 1, 1, 1, 1, 0],
              [0, 0, 1, 1, 3, 0, 0],
              [0, 0, 0, 1, 0, 0, 0]]

    body6 = [[0, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0],
              [0, 1, 3, 3, 1, 3, 1, 0],
              [1, 1, 1, 1, 1, 3, 1, 1],
              [1, 1, 3, 1, 1, 1, 1, 1],
              [0, 1, 3, 1, 3, 3, 1, 0],
              [0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 1, 1, 0, 0, 0]]

    body7 = [[0, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0],
              [0, 1, 3, 1, 1, 3, 1, 0],
              [1, 3, 1, 1, 3, 1, 1, 1],
              [1, 1, 1, 3, 1, 1, 3, 1],
              [0, 1, 3, 1, 1, 3, 1, 0],
              [0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 1, 1, 0, 0, 0]]

    pants1 = [[1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1,],
              [1, 1, 1, 1, 1, 1,],
              [1, 1, 1, 1, 1, 1,],
              [1, 1, 1, 1, 1, 1,]]

    pants2 = [[1, 1, 1, 1, 1, 1],
              [1, 3, 3, 1, 1, 1],
              [1, 1, 1, 1, 1, 1,],
              [1, 1, 1, 1, 1, 1,],
              [1, 1, 1, 3, 3, 1,],
              [1, 1, 1, 1, 1, 1,]]

    pants3 = [[1, 1, 1, 1, 1, 1],
              [1, 3, 1, 1, 1, 1],
              [1, 1, 1, 3, 1, 1,],
              [1, 1, 3, 1, 1, 1,],
              [1, 1, 1, 1, 3, 1,],
              [1, 1, 1, 1, 1, 1,]]

    pants4 = [[1, 1, 1, 1, 1, 1, 1],
              [1, 1, 3, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 3, 1],
              [1, 1, 1, 3, 1, 1, 1],
              [1, 3, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 3, 1, 1],
             [1, 1, 1, 1, 1, 1, 1]]

    pants5 = [[1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 3, 1, 3, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1],
                 [1, 3, 1, 1, 1, 3, 1],
                 [1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 3, 1, 3, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1]]


    pants6 = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, 3, 1, 1, 1, 1, 3, 1],
              [1, 1, 3, 1, 1, 3, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 3, 1, 1, 3, 1, 1],
              [1, 3, 1, 1, 1, 1, 3, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

    pants7 = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, 3, 1, 1, 1, 1, 3, 1],
              [1, 1, 3, 1, 3, 3, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 3, 3, 1, 3, 1, 1],
              [1, 3, 1, 1, 1, 1, 3, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

    gloves1 = [[0, 1, 1, 1, 1, 0],
             [1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1],
             [0, 1, 1, 1, 1, 0]]

    gloves2 = [[0, 1, 1, 1, 1, 0],
             [1, 1, 1, 3, 1, 1],
             [1, 1, 1, 1, 3, 1],
             [1, 3, 1, 1, 1, 1],
             [1, 1, 3, 1, 1, 1],
             [0, 1, 1, 1, 1, 0]]

    gloves3 = [[0, 1, 1, 1, 1, 0],
             [1, 3, 1, 3, 1, 1],
             [1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1],
             [1, 1, 3, 1, 3, 1],
             [0, 1, 1, 1, 1, 0]]

    gloves4 = [[0, 0, 1, 1, 1, 0, 0],
              [0, 1, 1, 1, 1, 1, 0],
              [1, 1, 3, 1, 3, 1, 1],
              [1, 1, 1, 3, 1, 1, 1],
              [1, 1, 3, 1, 3, 1, 1],
              [0, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 0, 0]]

    gloves5 = [[0, 0, 1, 1, 1, 0, 0],
              [0, 1, 1, 3, 1, 1, 0],
              [1, 1, 1, 1, 3, 1, 1],
              [1, 3, 1, 1, 1, 3, 1],
              [1, 1, 3, 1, 1, 1, 1],
              [0, 1, 1, 3, 1, 1, 0],
             [0, 0, 1, 1, 1, 0, 0]]

    gloves6 = [[0, 0, 1, 1, 1, 1, 0, 0],
              [0, 1, 1, 1, 1, 1, 1, 0],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 3, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 1, 1, 0],
              [0, 0, 1, 1, 1, 1, 0, 0]]

    gloves7 = [[0, 0, 1, 1, 1, 1, 0, 0],
              [0, 1, 3, 1, 1, 3, 1, 0],
              [1, 3, 1, 1, 1, 1, 3, 1],
              [1, 1, 1, 3, 1, 1, 1, 1],
              [1, 1, 1, 1, 3, 1, 1, 1],
              [1, 3, 1, 1, 1, 1, 3, 1],
              [0, 1, 3, 1, 1, 3, 1, 0],
              [0, 0, 1, 1, 1, 1, 0, 0]]

    weapon1 = [[0, 0, 1, 1, 0, 0],
             [0, 1, 3, 1, 1, 0],
             [1, 1, 3, 1, 1, 1, ],
             [1, 1, 1, 3, 1, 1, ],
             [0, 1, 1, 3, 1, 0, ],
             [0, 0, 1, 1, 0, 0, ]]

    weapon2 = [[0, 0, 1, 1, 0, 0],
             [0, 1, 3, 1, 1, 0],
             [1, 3, 1, 1, 1, 1, ],
             [1, 1, 1, 1, 3, 1, ],
             [0, 1, 1, 3, 1, 0, ],
             [0, 0, 1, 1, 0, 0, ]]

    weapon3 = [[0, 0, 1, 1, 0, 0],
             [0, 1, 1, 3, 1, 0],
             [1, 3, 1, 1, 1, 1, ],
             [1, 1, 1, 1, 3, 1, ],
             [0, 1, 3, 1, 1, 0, ],
             [0, 0, 1, 1, 0, 0, ]]

    weapon4 =[[0, 0, 0, 1, 0, 0, 0],
              [0, 0, 1, 1, 3, 0, 0],
              [0, 3, 1, 3, 1, 1, 0],
              [1, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 3, 1, 3, 0],
              [0, 0, 3, 1, 1, 0, 0],
              [0, 0, 0, 1, 0, 0, 0]]

    weapon5 =[[0, 0, 0, 1, 0, 0, 0],
              [0, 0, 1, 1, 3, 0, 0],
              [0, 3, 1, 1, 1, 1, 0],
              [1, 1, 1, 3, 1, 1, 1],
              [0, 1, 1, 1, 1, 3, 0],
              [0, 0, 3, 1, 1, 0, 0],
              [0, 0, 0, 1, 0, 0, 0]]

    weapon6 =[[0, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 1, 1, 3, 1, 0, 0],
              [0, 1, 3, 1, 1, 3, 1, 0],
              [1, 1, 1, 1, 1, 3, 1, 1],
              [1, 1, 3, 1, 1, 1, 1, 1],
              [0, 1, 3, 1, 1, 3, 1, 0],
              [0, 0, 1, 3, 1, 1, 0, 0],
              [0, 0, 0, 1, 1, 0, 0, 0]]

    weapon7 =[[0, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 1, 3, 1, 1, 0, 0],
              [0, 1, 1, 1, 3, 1, 1, 0],
              [1, 3, 1, 3, 1, 1, 1, 1],
              [1, 1, 1, 1, 3, 1, 3, 1],
              [0, 1, 1, 3, 1, 1, 1, 0],
              [0, 0, 1, 1, 3, 1, 0, 0],
              [0, 0, 0, 1, 1, 0, 0, 0]]

    turn_list_head = [7, 7, 7, 8, 8, 11, 11]
    turn_list_shoulder = [7, 7, 7, 8, 8, 11, 11]
    turn_list_body = [5, 5, 5, 5, 5, 8, 8]
    turn_list_pants = [7, 7, 7, 8, 8, 11, 11]
    turn_list_gloves = [7, 7, 7, 8, 8, 11, 11]
    turn_list_weapon = [5, 5, 5, 5, 5, 8, 8]
    stage_head_list = [head1, head2, head3, head4, head5, head6, head7]
    stage_shoulder_list = [shoulder1, shoulder2, shoulder3, shoulder4, shoulder5, shoulder6, shoulder7]
    stage_body_list = [body1, body2, body3, body4, body5, body6, body7]
    stage_pants_list = [pants1, pants2, pants3, pants4, pants5, pants6, pants7]
    stage_gloves_list = [gloves1, gloves2, gloves3, gloves4, gloves5, gloves6, gloves7]
    stage_weapon_list = [weapon1, weapon2, weapon3, weapon4, weapon5, weapon6, weapon7]

    stage_list = [stage_head_list, stage_shoulder_list, stage_body_list, stage_pants_list, stage_gloves_list, stage_weapon_list]
    stage_dict = {'head': stage_head_list, 'shoulder': stage_shoulder_list, 'body': stage_body_list,
                  'pants': stage_pants_list, 'gloves': stage_gloves_list, 'weapon': stage_weapon_list}
    turn_dict = {'head': turn_list_head, 'shoulder': turn_list_shoulder, 'body': turn_list_body,
                 'pants': turn_list_pants, 'gloves': turn_list_gloves, 'weapon': turn_list_weapon}
    stage_type_list = ['head', 'shoulder', 'body', 'pants', 'gloves', 'weapon']

    @staticmethod
    def get_stage_number(stage_name):
        for i, stg_name in enumerate(Stage.stage_type_list):
            if stage_name == stg_name:
                return i
        return -1


def get_map_with_turn(stage_name, stage_level=0, game_num=1):
    map = list()
    turn = list()
    for i in range(game_num):
        map.append(deepcopy(Stage.stage_dict[stage_name][stage_level]))
        turn.append(deepcopy(Stage.turn_dict[stage_name][stage_level]))
    return map, turn


def get_random_map_with_turn(game_num):
    map = list()
    turn = list()
    for i in range(game_num):
        stage_type = random.choice(Stage.stage_type_list)
        stage_num = random.randint(0, len(Stage.stage_dict[stage_type]) - 1)
        map.append(deepcopy(Stage.stage_dict[stage_type][stage_num]))
        turn.append(deepcopy(Stage.turn_dict[stage_type][stage_num]))
    return map, turn


def get_all_maps(game_num):
    map = list()
    turn = list()

    while(len(map) < game_num):
        stage_type = random.choice(Stage.stage_type_list)
        stage_num = random.randint(0, len(Stage.stage_dict[stage_type]) - 1)
        map.append(Stage.stage_dict[stage_type][stage_num])
        turn.append(Stage.turn_dict[stage_type][stage_num])


def get_turn(stage_num, level=0):
    return Stage.turn_dict[stage_num][level]


class KeyHolder(object):
    COMM_OP = 'op'
    COMM_STEP = 'step'
    COMM_INIT = 'init'
    COMM_RESET = 'reset'

    MY_TURN = 'my_turn'
    OPP_TURN = 'opp_turn'
    TOTAL_TURN = 'total_turn'
    GAME_RESULT = 'game_result'

    GAME_END_TURN = 'game_end_turn'

    TURN_LIMIT = 'turn_limit'

    VERSION_DECK_SIZE = 'version_deck_size'

    GAME_INFO_EPISODE_REWARD = 'episode_reward'
    GAME_INFO_FINE_EPISODE_REWARD = 'fine_episode_reward'
    REWARD = 'reward'
    FINE_REWARD = 'fine_reward'

    SIMULATOR_SEED = 'seed'

    STAGE_ID = 'stage_id'
    # head, shoulder, body, pants, gloves
    STAGE_NAME = 'stage_name'

    # 0~6
    STAGE_LEVEL = 'stage_level'

    NEW_UPDATE_CHAR_ID = 'new_update_char_id'
    ELZOWIN_LEVEL = 'elzowin_level'
    MAX_ELZOWIN_LEVEL = 'max_elzowin_level'
    MAX_TURN = 'max_turn'

    SIMULATOR_CONFIG = 'sim_config'
    SIMULATOR_ENV_SIZE = 'env_size'

    TOTAL_ENV_NUM = 'total_env_num'
    SIMULATOR_NUM = 'simulator_num'
    SIMULATOR_SETTING_LIST = 'setting_list'
    SAVE_REPLAY = 'save_replay'

    STAGE_HEAD = 'head'
    STAGE_SHOULDER = 'shoulder'
    STAGE_BODY = 'body'
    STAGE_PANTS = 'pants'
    STAGE_GLOVES = 'gloves'
    STAGE_RANDOM = 'random'

    SUB_GROUP_MASK_POSITION = 'make_position'
    SUB_GROUP_MASK_SELECT_CARD = 'select_card'
    SUB_GROUP_MASK_USE_REROLL = 'use_reroll'