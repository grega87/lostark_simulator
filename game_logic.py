import copy
from stage import *
import numpy as np


class TranscendGame:
    def __init__(self, show_card_debug=False):
        self.game_solved_count = 0
        self.show_card_debug = show_card_debug

        self.sim_config = None

        self.map = list()
        self.turn = list()
        self.elzowin_level = list()

        self.game_num = None
        self.game_result = None

        self.save_record = False

        self.map_ch_size = 10
        self.map_max_width_height = 8

        self.placeable_blocks = list()
        self.first_turn = list()
        self.card_list = list()
        self.top_deck = list()
        self.card_level_list = list()
        self.reroll_list = list()
        self.block_flag = list()
        self.spawn_flag = list()
        self.spawned_num = list()
        self.last_used_card = list()
        self.last_used_card_level = list()
        self.destroyed_block_list = list()
        self.replay_text = list()
        self.is_apply_block_flag = list()

        self.step_result_array = list()
        self.game_result_array = list()

        self.first_map = list()
        self.total_blocks = list()

        self.max_turn = list()

    def log(self, game_num=1):
        print("------------------log------------------")
        print("map : ", self.map[game_num])
        print("turn : ", self.turn[game_num])
        print("card_list : ", self.card_list[game_num], self.top_deck[game_num])
        print("card_level_list : ", self.card_level_list[game_num])
        print("reroll_list : ", self.reroll_list[game_num])
        print("block_flag : ", self.block_flag[game_num])
        print("spawn_flag : ", self.spawn_flag[game_num])
        print("last_used_card : ", self.last_used_card[game_num])
        print("last_used_card_level : ", self.last_used_card_level[game_num])
        print("destroyed_block_list : ", self.destroyed_block_list[game_num])

    def get_destroyable_blocks(self, game_num):
        return_list = list()
        for i in range(len(self.map[game_num])):
            for j in range(len(self.map[game_num][0])):
                if self.map[game_num][i][j] in BlockDefine.BREAKABLE.value:
                    return_list.append([i, j])
        return return_list

    def reset(self, config):
        self.map = list()
        self.turn = list()
        self.elzowin_level = list()
        self.placeable_blocks = list()

        self.sim_config = config[KeyHolder.SIMULATOR_CONFIG]
        self.game_num = self.sim_config[KeyHolder.SIMULATOR_ENV_SIZE]
        assert self.game_num == len(self.sim_config[KeyHolder.SIMULATOR_SETTING_LIST])

        self.save_record = self.sim_config[KeyHolder.SAVE_REPLAY]
        self.max_turn = list()

        # game_result에 스테이지 아이디 / 레벨 / 엘조윈 레벨 값 저장
        for each_config in self.sim_config[KeyHolder.SIMULATOR_SETTING_LIST]:
            stage_type = each_config[KeyHolder.STAGE_NAME]
            if stage_type == KeyHolder.STAGE_RANDOM:
                stage_type = random.choice(Stage.stage_type_list)

            stage_level = each_config[KeyHolder.STAGE_LEVEL]
            if stage_level == -1: # 랜덤 스테이지로 설정되었다면
                stage_level = random.randint(0, len(Stage.stage_dict[stage_type]) - 1)

            self.map.append(deepcopy(Stage.stage_dict[stage_type][stage_level]))
            self.turn.append(deepcopy(Stage.turn_dict[stage_type][stage_level]))

            elzowin_level = each_config[KeyHolder.ELZOWIN_LEVEL]
            if elzowin_level == -1:
                elzowin_level = random.randint(0, self.sim_config[KeyHolder.MAX_ELZOWIN_LEVEL])
            self.elzowin_level.append(elzowin_level)
            if KeyHolder.MAX_TURN in each_config.keys() and each_config[KeyHolder.MAX_TURN] - self.turn[-1] > 0:
                self.max_turn.append(each_config[KeyHolder.MAX_TURN] - self.turn[-1])
            else:
                self.max_turn.append(0)

        self.reroll_list = [2 for i in range(self.game_num)]
        for g in range(self.game_num):
            if self.elzowin_level[g] > 0:
                self.add_elzowin_bless(g, self.elzowin_level[g])

        self.placeable_blocks = [[] for i in range(self.game_num)]
        self.total_blocks = [0 for i in range(self.game_num)]
        for g in range(self.game_num):
            for i in range(len(self.map[g])):
                for j in range(len(self.map[g][i])):
                    if (self.map[g][i][j] == BlockDefine.DESTROYABLE.value or self.map[g][i][j] ==
                            BlockDefine.DISTORTION.value):
                        self.placeable_blocks[g].append([i, j])
                    if self.map[g][i][j] == BlockDefine.DESTROYABLE.value:
                        self.total_blocks[g] += 1

        self.first_turn = copy.deepcopy(self.turn)
        self.game_done = [False for i in range(self.game_num)]
        self.game_result = [0 for i in range(self.game_num)]
        self.card_level_list = [[1, 1] for i in range(self.game_num)]
        self.card_list = [[CardDefine.NONE, CardDefine.NONE] for i in range(self.game_num)]
        self.top_deck = [[CardDefine.NONE, CardDefine.NONE, CardDefine.NONE] for i in range(self.game_num)]
        self.block_flag = [0 for i in range(self.game_num)]
        self.spawn_flag = [0 for i in range(self.game_num)]
        self.spawned_num = [0 for i in range(self.game_num)]
        self.last_used_card = [CardDefine.NONE for i in range(self.game_num)]
        self.last_used_card_level = [1 for i in range(self.game_num)]
        self.destroyed_block_list = [[] for i in range(self.game_num)]
        self.replay_text = [[] for i in range(self.game_num)]
        self.is_apply_block_flag = [False for i in range(self.game_num)]

        self.step_result_array = list()
        self.game_result_array = list()

        self.prob_map = torch.tensor(ProbByDistance[1:], dtype=torch.float32)
        self.move_step = torch.arange(1, self.prob_map.shape[0], dtype=torch.long)
        self.move_pos = self.move_step
        self.move_neg = -self.move_step

        self.full_prob_map = torch.tensor(ProbByDistance, dtype=torch.float32)
        self.full_move_step = torch.arange(0, self.full_prob_map.shape[0], dtype=torch.long)
        self.full_move_pos = self.full_move_step
        self.full_move_neg = -self.full_move_step

        for i in range(self.game_num):
            self.get_random_card(i)

        if self.save_record:
            self.save_replay()

        self.update_game_result()

        return self.get_data()

    def reset_game_flags(self):
        self.game_done = [False for i in range(self.game_num)]
        self.game_result = [0 for i in range(self.game_num)]

    def save_replay(self):
        with open("log.txt", "a") as log_file:
            for i in range(len(self.map[0])-1, -1, -1):
                for j in range(len(self.map[0][0])):
                    log_file.write(str(self.map[0][j][i]) + " ")
                log_file.write("\n")
            log_file.write(str(self.turn[0]) + "\n")
            log_file.write(str(CardDefine(self.card_list[0][0]).name) + " " + str(self.card_level_list[0][0]) + " ")
            log_file.write(str(CardDefine(self.card_list[0][1]).name) + " " + str(self.card_level_list[0][1]) + "\n")
            log_file.write(str(CardDefine(self.top_deck[0][0]).name) + " ")
            log_file.write(str(CardDefine(self.top_deck[0][1]).name) + " ")
            log_file.write(str(CardDefine(self.top_deck[0][2]).name) + "\n")
            if len(self.replay_text[0]) > 0:
                log_file.write(self.replay_text[0][-1] + "\n")
            else:
                log_file.write("game restart\n")
            log_file.write("=====================================\n")

        pass

    def step(self, action_list):
        self.update_map(action_list.selected_action)
        self.apply_block_flag()
        for i in range(self.game_num):
            self.get_random_card(i)
        self.check_game_result()

        self.update_game_result()

        # 게임이 끝난 것들만 게임 재시작... 설정
        for env_id in range(self.game_num):
            if self.game_done[env_id]:
                self.restart_game(env_id)

        if self.save_record:
            self.save_replay()

        return self.get_data()

    def get_random_card(self, game_num):
        if self.card_list[game_num][1] == self.card_list[game_num][0] and self.card_level_list[game_num][0] < 3 and\
        self.card_list[game_num][0] not in [CardDefine.NONE, CardDefine.ERUPTION.value, CardDefine.WORLD_TREE.value]:
            self.card_level_list[game_num][0] += 1
            self.card_list[game_num][1] = CardDefine.NONE
        while CardDefine.NONE in self.card_list[game_num] or CardDefine.NONE in self.top_deck[game_num]:
            # new_card = CardDefine.NONE
            card_index = np.random.choice(len(CardProbabilities), p=CardProbabilities)
            new_card = CardDefine(card_index + 1)
            if self.show_card_debug:
                print(f"처음 카드 : {self.card_list[game_num]}, 레벨 : {self.card_level_list[game_num]}")
                print(f"탑 덱 : {self.top_deck[game_num]}")
            if self.card_list[game_num][1] == CardDefine.NONE:
                if self.card_list[game_num][0] == CardDefine.NONE:
                    self.card_list[game_num][0] = new_card.value
                    # when no card is in the hand
                else:
                    if self.top_deck[game_num][0] != CardDefine.NONE:
                        if self.card_list[game_num][0] == self.top_deck[game_num][0] and self.card_level_list[game_num][0] < 3  and\
                        self.card_list[game_num][0] not in [CardDefine.NONE, CardDefine.ERUPTION.value, CardDefine.WORLD_TREE.value]:
                            self.card_level_list[game_num][0] += 1
                        else:
                            self.card_list[game_num][1] = self.top_deck[game_num][0]
                        self.top_deck[game_num][0] = self.top_deck[game_num][1]
                        self.top_deck[game_num][1] = self.top_deck[game_num][2]
                        self.top_deck[game_num][2] = new_card.value
                    elif self.card_list[game_num][0] == new_card.value and self.card_level_list[game_num][0] < 3 and\
                    self.card_list[game_num][0] not in [CardDefine.NONE, CardDefine.ERUPTION.value, CardDefine.WORLD_TREE.value]:
                        self.card_level_list[game_num][0] += 1
                    else:
                        self.card_list[game_num][1] = new_card.value
            elif self.card_list[game_num][0] == CardDefine.NONE:
                if self.card_list[game_num][1] == self.top_deck[game_num][0] and self.card_level_list[game_num][
                    1] < 3 and self.card_list[game_num][1] not in [CardDefine.NONE, CardDefine.ERUPTION.value, CardDefine.WORLD_TREE.value]:
                    self.card_level_list[game_num][1] += 1
                else:
                    self.card_list[game_num][0] = self.top_deck[game_num][0]
                self.top_deck[game_num][0] = self.top_deck[game_num][1]
                self.top_deck[game_num][1] = self.top_deck[game_num][2]
                self.top_deck[game_num][2] = new_card.value
                for i in range(3):
                    if self.top_deck[game_num][i] == CardDefine.NONE:
                        self.top_deck[game_num][i] = random.randint(1, CardDefine.TYPHOON.value)
                        break
            # when two cards are in the hand
            elif CardDefine.NONE not in self.card_list[game_num] and self.top_deck[game_num][0] == CardDefine.NONE:
                for i in range(3):
                    if self.top_deck[game_num][i] == CardDefine.NONE:
                        self.top_deck[game_num][i] = random.randint(1, CardDefine.TYPHOON.value)
                        break
            # when adding card to top of deck
            else:
                for i in range(3):
                    if self.top_deck[game_num][i] == CardDefine.NONE:
                        self.top_deck[game_num][i] = new_card.value
                        break
            if self.show_card_debug:
                print(f"끝 카드 : {self.card_list[game_num]}, 레벨 : {self.card_level_list[game_num]}")
                print(f"탑 덱 : {self.top_deck[game_num]}")
                print("=====================================")

    def check_game_result(self):
        for i in range(self.game_num):
            self.game_result[i] = 1
            self.game_done[i] = True
            block_num = 0
            for x in range(len(self.map[i])):
                for y in range(len(self.map[i][0])):
                    if self.map[i][x][y] in BlockDefine.BREAKABLE.value:
                        self.game_result[i] = 0
                        self.game_done[i] = False
                        block_num += 1
            #print(f"game {i} result : {self.game_result[i]} left blocks : {block_num}")
            if self.game_done[i] == True:
                if self.show_card_debug:
                    print("game solved! ", i)
                    print(self.turn[i])
                if self.turn[i] < 0:
                    self.game_result[i] = -1
                else:
                    self.game_solved_count += 1
            if self.turn[i] <= 0 - self.max_turn[i]:
                if self.show_card_debug:
                    print("reset, left blocks : ", block_num)
                self.game_result[i] = -1
                self.game_done[i] = True

    def apply_block_flag(self):
        for i in range(self.game_num):
            self.spawned_num[i] = 0
        for i in range(self.game_num):
            if not self.is_apply_block_flag[i]:
                continue
            if self.spawn_flag[i] > 0:
                block_candidates = list()
                for b in self.placeable_blocks[i]:
                    if self.map[i][b[0]][b[1]] == BlockDefine.EMPTY.value:
                        block_candidates.append(b)
                for n in range(self.spawn_flag[i]):
                    if len(block_candidates) == 0:
                        break
                    block = random.choice(block_candidates)
                    block_candidates.remove(block)
                    self.map[i][block[0]][block[1]] = BlockDefine.DESTROYABLE.value
                    self.spawned_num[i] += 1
                self.spawn_flag[i] = 0

            if self.block_flag[i] == BlockDefine.ADDITION.value:
                self.reroll_list[i] += 1
            elif self.block_flag[i] == BlockDefine.BLESS.value:
                self.turn[i] += 1
            elif self.block_flag[i] == BlockDefine.MYSTIC.value:
                # GET ERUPTION OR WORLD_TREE CARD
                new_card = CardDefine.WORLD_TREE.value
                if random.randint(0, 1) == 0:
                    new_card = CardDefine.ERUPTION.value
                if self.card_list[i][0] == CardDefine.NONE:
                    self.card_list[i][1] = new_card
                else:
                    self.card_list[i][0] = new_card
            elif self.block_flag[i] == BlockDefine.ENFORCE.value:
                if self.card_list[i][0] not in [CardDefine.NONE, CardDefine.ERUPTION.value, CardDefine.WORLD_TREE.value] \
                        and self.card_level_list[i][0] < 3:
                    self.card_level_list[i][0] += 1
                elif self.card_list[i][1] not in [CardDefine.NONE, CardDefine.ERUPTION.value, CardDefine.WORLD_TREE.value] \
                        and self.card_level_list[i][1] < 3:
                    self.card_level_list[i][1] += 1
            elif self.block_flag[i] == BlockDefine.COPY.value:
                if self.card_list[i][0] == CardDefine.NONE:
                    self.card_list[i][1] = self.last_used_card[i]
                    self.card_level_list[i][1] = self.last_used_card_level[i]
                elif self.card_list[i][1] == CardDefine.NONE:
                    self.card_list[i][0] = self.last_used_card[i]
                    self.card_level_list[i][0] = self.last_used_card_level[i]
            elif self.block_flag[i] == BlockDefine.RELOCATION.value:
                # move all BLOCKS INTO A RANDOM POSITION
                # first get all number of blocks
                destroyable_block_num = 0
                disorted_block_num = 0
                for x in range(len(self.map[i])):
                    for y in range(len(self.map[i][0])):
                        if self.map[i][x][y] == BlockDefine.DESTROYABLE.value:
                            destroyable_block_num += 1
                        elif self.map[i][x][y] == BlockDefine.DISTORTION.value:
                            disorted_block_num += 1
                # clean map
                for x in range(len(self.map[i])):
                    for y in range(len(self.map[i][0])):
                        self.map[i][x][y] = BlockDefine.EMPTY.value
                # replace blocks
                block_candidates = copy.deepcopy(self.placeable_blocks[i])
                for n in range(destroyable_block_num + disorted_block_num):
                    block = random.choice(block_candidates)
                    block_candidates.remove(block)
                    if n < destroyable_block_num:
                        self.map[i][block[0]][block[1]] = BlockDefine.DESTROYABLE.value
                    else:
                        self.map[i][block[0]][block[1]] = BlockDefine.DISTORTION.value
        for i in range(self.game_num):
            if not self.is_apply_block_flag[i]:
                continue
            block_candidates = list()
            for j in range(len(self.map[i])):
                for k in range(len(self.map[i][j])):
                    if (self.map[i][j][k] > BlockDefine.DESTROYED.value and
                            self.map[i][j][k] != BlockDefine.DISTORTION.value):
                        self.map[i][j][k] = BlockDefine.DESTROYABLE.value
                    if self.map[i][j][k] == BlockDefine.DESTROYABLE.value:
                        block_candidates.append([i, j, k])
            random_num = random.random()
            prob_sum = 0
            new_block = None
            for b, v in enumerate(BlockProbabilities):
                prob_sum += v
                if random_num < prob_sum:
                    new_block = BlockList[b]
                    break
            if len(block_candidates) == 0:
                continue
            pos = random.choice(block_candidates)
            self.map[i][pos[1]][pos[2]] = new_block
        for i in range(self.game_num):
            self.block_flag[i] = 0
            self.is_apply_block_flag[i] = False

    def restart_game(self, env_id):
        sim_settings = self.sim_config[KeyHolder.SIMULATOR_SETTING_LIST]

        self.card_list[env_id] = [CardDefine.NONE, CardDefine.NONE]
        self.top_deck[env_id] = [CardDefine.NONE, CardDefine.NONE, CardDefine.NONE]
        self.card_level_list[env_id] = [1, 1]
        self.reroll_list[env_id] = 2
        self.block_flag[env_id] = 0
        self.spawn_flag[env_id] = 0
        self.spawned_num[env_id] = 0
        self.get_random_card(env_id)
        self.destroyed_block_list[env_id] = list()
        self.last_used_card[env_id] = CardDefine.NONE
        self.last_used_card_level[env_id] = 1

        each_config = sim_settings[env_id]
        stage_type = each_config[KeyHolder.STAGE_NAME]
        if stage_type == KeyHolder.STAGE_RANDOM:
            stage_type = random.choice(Stage.stage_type_list)

        stage_level = each_config[KeyHolder.STAGE_LEVEL]
        if stage_level == -1:  # 랜덤 스테이지로 설정되었다면
            stage_level = random.randint(0, len(Stage.stage_dict[stage_type]) - 1)

        self.map[env_id] = deepcopy(Stage.stage_dict[stage_type][stage_level])
        self.turn[env_id] = deepcopy(Stage.turn_dict[stage_type][stage_level])

        self.first_turn[env_id] = copy.deepcopy(self.turn[env_id])
        self.replay_text = [[] for i in range(self.game_num)]

        elzowin_level = each_config[KeyHolder.ELZOWIN_LEVEL]
        if elzowin_level == -1:
            elzowin_level = random.randint(0, self.sim_config[KeyHolder.MAX_ELZOWIN_LEVEL])
        self.elzowin_level[env_id] = elzowin_level

        if elzowin_level > 0:
            self.add_elzowin_bless(elzowin_level)
        self.placeable_blocks[env_id] = []
        self.total_blocks[env_id] = 0
        for i in range(len(self.map[env_id])):
            for j in range(len(self.map[env_id][0])):
                if (self.map[env_id][i][j] == BlockDefine.DESTROYABLE.value or self.map[env_id][i][j] ==
                        BlockDefine.DISTORTION.value):
                    self.placeable_blocks[env_id].append([i, j])
                if self.map[env_id][i][j] == BlockDefine.DESTROYABLE.value:
                    self.total_blocks[env_id] += 1


    def add_elzowin_bless(self, env_id, level=1):
        distortion_block_list = list()
        for x in range(len(self.map[env_id][0])):
            for y in range(len(self.map[env_id][1])):
                if self.map[env_id][x][y] == BlockDefine.DISTORTION.value:
                    distortion_block_list.append([x, y])

        # level만큼의 블럭 제거
        if len(distortion_block_list) > 0:
            for _ in range(level):
                block = random.choice(distortion_block_list)
                distortion_block_list.remove(block)
                self.map[env_id][block[0]][block[1]] = BlockDefine.DESTROYABLE.value

        # level 만큼 리롤 숫자 추가
        self.reroll_list[env_id] += level

    def change_card(self, game_num, card_num):
        self.card_list[game_num][card_num] = CardDefine.NONE
        self.card_level_list[game_num][card_num] = 1
        self.get_random_card(game_num)

    def update_map(self, action_list):
        use_reroll = action_list[KeyHolder.SUB_GROUP_MASK_USE_REROLL]
        select_card = action_list[KeyHolder.SUB_GROUP_MASK_SELECT_CARD]
        make_position = action_list[KeyHolder.SUB_GROUP_MASK_POSITION]

        for i in range(self.game_num):
            each_use_reroll = use_reroll[i].item()
            if each_use_reroll > 0:
                self.change_card(i, each_use_reroll - 1)
                self.replay_text[i].append(f"reroll {each_use_reroll}")
                self.reroll_list[i] -= 1
            else:
                each_select_card = int(select_card[i])
                each_pos = make_position[i]
                card_num = each_select_card - 1 if each_select_card > 0 else 0

                x = int(each_pos // 8)
                y = int(each_pos % 8)

                card_name = CardDefine(self.card_list[i][card_num]).name
                level = self.card_level_list[i][card_num]
                self.replay_text[i].append(f"card_num {card_num} use {card_name} level {level} at {x}, {y}")
                self.use_card(i, card_num, x, y)
                self.is_apply_block_flag[i] = True

    def check_block_valid(self, game_num, x, y):
        if x < 0 or y < 0 or x >= len(self.map[game_num]) or y >= len(self.map[game_num][0]):
            return False
        return True

    def break_block(self, game_num, x, y, percent, can_destroy_distortion, can_avoid_distortion):
        #print(f"break block {x}, {y} env {game_num} percent {percent} can destroy distortion {can_destroy_distortion} can avoid distortion {can_avoid_distortion}")
        if x not in range(len(self.map[game_num])) or y not in range(len(self.map[game_num][0])):
            print(f"Error , x : {x}, y : {y}")
            return
        if random.randint(0, 99) >= percent:
            return
        if self.map[game_num][x][y] in BlockDefine.BREAKABLE.value:
            if self.map[game_num][x][y] > 3:
                self.block_flag[game_num] = self.map[game_num][x][y]
            self.map[game_num][x][y] = BlockDefine.EMPTY.value
            self.destroyed_block_list[game_num].append([x, y])
        elif self.map[game_num][x][y] == BlockDefine.DISTORTION.value:
            if can_destroy_distortion:
                self.map[game_num][x][y] = BlockDefine.EMPTY.value
                self.destroyed_block_list[game_num].append([x, y])
            elif not can_destroy_distortion and not can_avoid_distortion:
                self.spawn_flag[game_num] += 3

    def use_card(self, game_num, card_num, x, y):
        self.last_used_card[game_num] = self.card_list[game_num][card_num]
        self.last_used_card_level[game_num] = self.card_level_list[game_num][card_num]
        # UP DOWN LEFT RIGHT 1 DISTANCE
        if self.card_list[game_num][card_num] == CardDefine.LIGHTNING.value:
            _x = [0, 0, -1, 1, 0]
            _y = [-1, 1, 0, 0, 0]
            for i in range(5):
                if self.check_block_valid(game_num, x + _x[i], y + _y[i]):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = 50
                        if _x[i] == 0 and _y[i] == 0:
                            percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, False, True)
        # UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT 1 DISTANCE
        if self.card_list[game_num][card_num] == CardDefine.TYPHOON.value:
            _x = [-1, -1, 1, 1, 0]
            _y = [-1, 1, -1, 1, 0]
            for i in range(5):
                if self.check_block_valid(game_num, x + _x[i], y + _y[i]):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = 50
                        if _x[i] == 0 and _y[i] == 0:
                            percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, False, True)
        # UP_LEFT, UP, UP_RIGHT, LEFT, RIGHT, DOWN_LEFT, DOWN, DOWN_RIGHT 1 DISTANCE
        if self.card_list[game_num][card_num] == CardDefine.EARTH.value:
            _x = [-1, -1, -1, 0, 0, 1, 1, 1, 0]
            _y = [-1, 0, 1, -1, 1, -1, 0, 1, 0]
            for i in range(9):
                if self.check_block_valid(game_num, x + _x[i], y + _y[i]):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = 75
                        if _x[i] == 0 and _y[i] == 0:
                            percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, False, True)
        # RIGHT LEFT 1 DISTANCE, LEVEL 3 ADDITIONAL UP DOWN 1 DISTANCE
        if self.card_list[game_num][card_num] == CardDefine.PURIFICATION.value:
            _x = [0, 1, -1, 0, 0]
            _y = [0, 0, 0, -1, 1]
            for i in range(5):
                if self.check_block_valid(game_num, x + _x[i], y + _y[i]):
                    if self.card_level_list[game_num][card_num] == 1 and i < 3:
                        percent = 50
                        if _x[i] == 0 and _y[i] == 0:
                            percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, True, False)
                    elif self.card_level_list[game_num][card_num] == 2 and i < 3:
                        percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, True, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, True, False)
        # 2DISTANCE ALL DIRECTIONS
        if self.card_list[game_num][card_num] == CardDefine.FIRE.value:
            _x = [0, 0, 0, 0, 1, 1, 1, -1, -1, -1, 2, -2, 0, 0, 0]
            _y = [-2, -1, 1, 2, -1, 0, 1, -1, 0, 1, 0, 0, 2, -2, 0]
            for i in range(15):
                if self.check_block_valid(game_num, x + _x[i], y + _y[i]):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = 50
                        if _x[i] == 0 and _y[i] == 0:
                            percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, x + _x[i], y + _y[i], percent, False, True)
        # LEFT RIGHT ALL BLOCKS
        if self.card_list[game_num][card_num] == CardDefine.EARTHQUAKE.value:
            for _x in range(7):
                if self.check_block_valid(game_num, _x + x, y):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = PercentByDistance[_x]
                        self.break_block(game_num, _x + x, y, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, _x + x, y, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, _x + x, y, percent, False, True)
                if self.check_block_valid(game_num, x - _x, y):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = PercentByDistance[_x]
                        self.break_block(game_num, x - _x, y, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, x - _x, y, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, x - _x, y, percent, False, True)
        # LEFT RIGHT UP DOWN ALL BLOCKS
        if self.card_list[game_num][card_num] == CardDefine.WATER.value:
            for _x in range(7):
                if self.check_block_valid(game_num, _x + x, y):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = PercentByDistance[_x]
                        self.break_block(game_num, _x + x, y, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, _x + x, y, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, _x + x, y, percent, False, True)
                if self.check_block_valid(game_num, x, y + _x):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = PercentByDistance[_x]
                        self.break_block(game_num, x, y + _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, x, y + _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, x, y + _x, percent, False, True)
                if self.check_block_valid(game_num, x, y - _x):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = PercentByDistance[_x]
                        self.break_block(game_num, x, y - _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, x, y - _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, x, y - _x, percent, False, True)
                if self.check_block_valid(game_num, x - _x, y):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = PercentByDistance[_x]
                        self.break_block(game_num, x - _x, y, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, x - _x, y, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, x - _x, y, percent, False, True)
        # UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT ALL BLOCKS
        if self.card_list[game_num][card_num] == CardDefine.EXPLOSION.value:
            for _x in range(7):
                if self.check_block_valid(game_num, _x + x, y + _x):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = PercentByDistance[_x]
                        self.break_block(game_num, _x + x, y + _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, _x + x, y + _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, _x + x, y + _x, percent, False, True)
                if self.check_block_valid(game_num, _x + x, y - _x):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = PercentByDistance[_x]
                        self.break_block(game_num, _x + x, y - _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, _x + x, y - _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, _x + x, y - _x, percent, False, True)
                if self.check_block_valid(game_num, x - _x, y + _x):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = PercentByDistance[_x]
                        self.break_block(game_num, x - _x, y + _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, x - _x, y + _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, x - _x, y + _x, percent, False, True)
                if self.check_block_valid(game_num, x - _x, y - _x):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = PercentByDistance[_x]
                        self.break_block(game_num, x - _x, y - _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, x - _x, y - _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, x - _x, y - _x, percent, False, True)
        # UP DOWN ALL BLOCKS
        if self.card_list[game_num][card_num] == CardDefine.WATERSPOUT.value:
            for _x in range(7):
                if self.check_block_valid(game_num, x, y + _x):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = PercentByDistance[_x]
                        self.break_block(game_num, x, y + _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, x, y + _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, x, y + _x, percent, False, True)
                if self.check_block_valid(game_num, x, y - _x):
                    if self.card_level_list[game_num][card_num] == 1:
                        percent = PercentByDistance[_x]
                        self.break_block(game_num, x, y - _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 2:
                        percent = 100
                        self.break_block(game_num, x, y - _x, percent, False, False)
                    elif self.card_level_list[game_num][card_num] == 3:
                        percent = 100
                        self.break_block(game_num, x, y - _x, percent, False, True)
        # -1 ~ 2 * level random blocks, -1 = make a random destroyed block
        if self.card_list[game_num][card_num] == CardDefine.THUNDER.value:
            break_block_num = random.randint(-1, self.card_level_list[game_num][card_num] * 2)
            self.break_block(game_num, x, y, 100, False, False)
            block_list = self.get_destroyable_blocks(game_num)
            if break_block_num == -1:
                self.spawn_flag[game_num] += 1
            else:
                for i in range(break_block_num):
                    if len(block_list) == 0:
                        break
                    block = random.choice(block_list)
                    self.break_block(game_num, block[0], block[1], 100, False, False)
                    block_list.remove(block)
        # JUST ONE BLOCK, NO LEVELS
        if self.card_list[game_num][card_num] == CardDefine.ERUPTION.value:
            self.break_block(game_num, x, y, 100, False, False)
        # RIGHT LEFT UP DOWN 2DISTANCE NO LEVELS
        if self.card_list[game_num][card_num] == CardDefine.WORLD_TREE.value:
            _x = [0, 0, 0, 0, -2, -1, 1, 2, 0]
            _y = [-2, -1, 1, 2, 0, 0, 0, 0, 0]
            for i in range(9):
                if self.check_block_valid(game_num, x + _x[i], y + _y[i]):
                    percent = 100
                    self.break_block(game_num, x + _x[i], y + _y[i], percent, True, False)
        self.card_level_list[game_num][card_num] = 1
        self.card_list[game_num][card_num] = CardDefine.NONE
        self.turn[game_num] -= 1

    def get_available_actions(self):
        mask = dict()
        mask['select_card'] = list()
        mask['use_reroll'] = list()
        mask['make_position'] = list()

        for g in range(len(self.map)):
            mask['make_position'].append([])

            # [None, USE_CARD_1, USE_CARD_2]
            mask['select_card'].append([1, 1, 1])

            # [None, CHANGE_CARD_1, CHANGE_CARD_2]
            if self.reroll_list[g] > 0:
                mask['use_reroll'].append([1, 1, 1])
            else:
                mask['use_reroll'].append([1, 0, 0])

            for c in range(2):
                mask['make_position'][g].append([])
                for i in range(self.map_max_width_height):
                    for j in range(self.map_max_width_height):
                        if i >= len(self.map[g]) or j >= len(self.map[g][0]):
                            mask['make_position'][g][c].append(0)
                        elif (self.card_list[g][c] in CardDefine.SPECIAL_CARD.value and self.map[g][i][j] in
                              (BlockDefine.BREAKABLE.value + [BlockDefine.DISTORTION.value])):
                            mask['make_position'][g][c].append(1)
                        elif self.map[g][i][j] in BlockDefine.BREAKABLE.value:
                            mask['make_position'][g][c].append(1)
                        else:
                            mask['make_position'][g][c].append(0)
        for k, v in mask.items():
            mask[k] = torch.tensor(v)
        return mask

    def update_game_result(self):
        self.step_result_array = list()
        # num blocks destroy, num blocks added, left blocks percentage
        for game_num in range(self.game_num):
            step_result_obs = list()
            step_result_obs.append(len(self.destroyed_block_list[game_num]))
            step_result_obs.append(self.spawned_num[game_num])
            num_blocks = 0
            for i in range(len(self.map[game_num][0])):
                for j in range(len(self.map[game_num][1])):
                    if self.map[game_num][i][j] in BlockDefine.BREAKABLE.value:
                        num_blocks += 1

            step_result_obs.append(num_blocks / len(self.placeable_blocks) if self.placeable_blocks else 0)
            self.step_result_array.append(step_result_obs)
            self.destroyed_block_list[game_num] = list()

        self.game_result_array = list()
        # total turns, left turns, done, result
        for game_num in range(self.game_num):
            game_result_obs = list()
            game_result_obs.append(self.turn[game_num])
            game_result_obs.append(self.first_turn[game_num] - self.turn[game_num])
            game_result_obs.append(not self.game_done[game_num])
            game_result_obs.append(self.game_result[game_num])

            env_game_setting = self.sim_config[KeyHolder.SIMULATOR_SETTING_LIST]
            game_result_obs.append(env_game_setting[game_num][KeyHolder.STAGE_LEVEL])
            stage_name = env_game_setting[game_num][KeyHolder.STAGE_NAME]
            for stage_number in range(len(Stage.stage_type_list)):
                if stage_name == Stage.stage_type_list[stage_number]:
                    game_result_obs.append(stage_number + 1)
                    break
            game_result_obs.append(env_game_setting[game_num][KeyHolder.ELZOWIN_LEVEL])
            self.game_result_array.append(game_result_obs)

    def get_data(self):
        obs = dict()

        env_size = len(self.map)
        tensor_map = torch.zeros((env_size, self.map_ch_size, self.map_max_width_height, self.map_max_width_height), dtype=torch.float32)
        for game_num in range(len(self.map)):
            map_obs = list()
            pos_y_size = len(self.map[game_num])
            pos_x_size = len(self.map[game_num][0])
            for i in range(BlockDefine.RELOCATION.value + 1):
                new_channel = list()
                for y in range(pos_y_size):
                    new_block = list()
                    for x in range(pos_x_size):
                        if i == self.map[game_num][y][x]:
                            new_block.append(1)
                        else:
                            new_block.append(0)
                    new_channel.append(new_block)
                map_obs.append(new_channel)
            # map_array.append(map_obs)
            tensor_map[game_num, :, :pos_y_size, :pos_x_size] = torch.tensor(map_obs, dtype=torch.float32)

        common_array = list()
        # total turn, left turn, reroll_number, curr_card1~2, next card1~3, card1~2 level
        env_game_setting = self.sim_config[KeyHolder.SIMULATOR_SETTING_LIST]
        for game_num in range(len(self.map)):
            common_obs = list()
            common_obs.append(self.turn[game_num])
            common_obs.append(self.first_turn[game_num] - self.turn[game_num])
            common_obs.append(self.reroll_list[game_num])
            common_obs.append(self.card_list[game_num][0])
            common_obs.append(self.card_list[game_num][1])
            common_obs.append(self.top_deck[game_num][0])
            common_obs.append(self.top_deck[game_num][1])
            common_obs.append(self.top_deck[game_num][2])
            common_obs.append(self.card_level_list[game_num][0])
            common_obs.append(self.card_level_list[game_num][1])
            common_obs.append(env_game_setting[game_num][KeyHolder.STAGE_LEVEL])
            stage_name = env_game_setting[game_num][KeyHolder.STAGE_NAME]
            for stage_number in range(len(Stage.stage_type_list)):
                if stage_name == Stage.stage_type_list[stage_number]:
                    common_obs.append(stage_number + 1)
                    break
            common_obs.append(env_game_setting[game_num][KeyHolder.ELZOWIN_LEVEL])
            common_array.append(common_obs)

        obs['map'] = tensor_map
        obs['common'] = torch.tensor(common_array)
        obs['step_result'] = torch.tensor(self.step_result_array)
        obs['game_result'] = torch.tensor(self.game_result_array)
        mask = self.get_available_actions()

        return {'obs': obs, 'mask': mask}
