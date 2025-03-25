from game_logic import TranscendGame
from gui.gui import GameSimulatorGUI, DualGameSimulatorGUI
import numpy as np
import tkinter as tk
import logging
import argparse
import os
import pickle

simula_config = {
    'seed': -1,  # SEED 값
    'env_size': 1,  # 동시 실행 게임 수
    'save_replay': False,
    "sim_config": {
        'env_size': 1,  # 동시 실행 게임 수
        'max_elzowin_level': 1,  # 엘조윈 레벨
        'save_replay': False,
        'setting_list': [ # max_turn은 실제로 진행 하는 최대 턴, 그 전에 끝나면 리셋
            {'stage_name': 'body', 'stage_level': 0, 'elzowin_level': 0, 'max_turn': 15} for x in range(1)]
    }
}


class return_box:
    def __init__(self):
        self.selected_action = dict()


def init_logger(
        log_level=logging.DEBUG,
):
    logging.basicConfig(
        format='\033[33m %(levelname)s \033[0m (%(asctime)s):\033[33m %(message)s \033[0m',
        level=log_level,
        datefmt='%m/%d/%Y %H:%M:%S %p'
    )


class RandomAgent(object):
    @staticmethod
    def select_action(info, data_spec):
        # 실제 수행은 전달된 데이터의 obs 와 result 데이터를 활용해 학습하고,
        # mask 데이터를 참고해 가능한 행동 중 하나를 선택하는 것이지만
        # 예제에서는 mask 정보만을 참고하여 선택 가능한 행동 중 하나를 선택하는 것으로 전송

        env_size = list(info['mask'].values())[0].shape[0]
        sel_act_prob = dict()
        for act, data in info['mask'].items():
            if act == 'make_position':
                prob_list = list()
                for i in range(2):
                    prob_list.append(data[0][i]/data.sum(axis=-1, keepdims=True))
                sel_act_prob[act] = np.stack(prob_list, axis=-1)
            sel_act_prob[act] = data / data.sum(axis=-1, keepdims=True)

        # 리턴 타입에 맞추어 Numpy 형태의 데이터 생성
        result = return_box()
        use_reroll = [False for x in range(env_size)]
        select_card = [0 for x in range(env_size)]
        # 선택 가능한 행동 중에서 랜덤 선택 수행
        for return_key in data_spec['action']['return_action']:
            if return_key == 'make_position':
                np_sel_act = np.zeros([env_size], dtype=np.int32)
                for env_id in range(env_size):
                    if not use_reroll[env_id]:
                        prob_array = sel_act_prob[return_key][env_id][select_card[env_id]-1].numpy() / sel_act_prob[return_key][env_id][select_card[env_id]-1].sum().numpy()
                        pos = int(np.random.choice(sel_act_prob[return_key][env_id][select_card[env_id]-1].shape[-1], 1,
                                                                  p=prob_array)[0])
                        np_sel_act[env_id] = pos
                result.selected_action[return_key] = np_sel_act
            else:
                act_prob = sel_act_prob[return_key]

                np_sel_act = np.zeros(env_size, dtype=np.int32)
                for env_id in range(env_size):
                    prob_array = act_prob[env_id].numpy() / act_prob[env_id].sum().numpy()
                    sel_raw_act_id = int(np.random.choice(act_prob.shape[-1], 1, p=prob_array)[0])
                    np_sel_act[env_id] = sel_raw_act_id
                    if return_key == 'use_reroll':
                        if sel_raw_act_id == 1:
                            use_reroll[env_id] = True
                    if return_key == 'select_card':
                        if use_reroll[env_id] == False:
                            act_prob[env_id][0] = 0
                            act_prob[env_id] = act_prob[env_id] / act_prob[env_id].sum(axis=-1, keepdims=True)
                            sel_raw_act_id = int(np.random.choice(act_prob.shape[-1], 1, p=act_prob[env_id])[0])
                            np_sel_act[env_id] = sel_raw_act_id
                        select_card[env_id] = sel_raw_act_id

                result.selected_action[return_key] = [np_sel_act]
        return result


# Encoding Data : Agent -> Simulator
def make_return_data(cmd, sim_config=None, result_data=None, show_result=True, prefix_tag=''):
    """
    :param cmd: {'init', 'reset', 'step'} 명령 중 하나. 최초 시뮬레이터 구동 시에는 init 명령, 실행 중간에 설정 변경이 필요한 경우 reset, 그 외 상황에서는 step
    :param sim_config: 시뮬레이터 동작을 위한 설정 정보들
    :param result_data: Agent 가 선택한 행동 정보들
    :param show_result: Agent 가 선택한 행동 정보를 표시할 지의 여부
    :param prefix_tag : 앞 단에 추가할 시작 채크
    :return: 통신을 위한 변환 정보
    """

    if show_result:
        if sim_config is not None:
            logging.info('>> {} : {}'.format(cmd.upper(), sim_config))

        if result_data is not None:
            logging.info('>> {}{} : {}'.format(prefix_tag, cmd.upper(), result_data))

    return {'cmd': cmd, 'sim_config': sim_config, 'result_data': result_data}


def run_test(args, root=None, moves=50):
    init_logger(log_level=logging.DEBUG)

    data_spec = None
    with open('data_spec.pkl', 'rb') as f:
        data_spec = pickle.load(f)
    simulator = TranscendGame(show_card_debug=True)
    simulator.reset(simula_config)

    player_simulator = TranscendGame(show_card_debug=True)
    player_simulator.reset(simula_config)

    # simulator.card_list[0] = [CardDefine.WORLD_TREE.value, CardDefine.THUNDER.value]
    # simulator.card_level_list[0] = [1, 1]
    recv_data = simulator.get_data()
    player_recv_data = player_simulator.get_data()

    if args.use_gui:
        # GUI 생성
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(current_dir, "gui")
        if args.dual_mode:
            # Dual GUI 모드 (플레이어 & AI 동시 진행)
            player_data = player_simulator  # Player Simulator 데이터
            agent_data = simulator  # Agent Simulator 데이터
            gui = DualGameSimulatorGUI(root, player_data, agent_data, base_path)
        else:
            if not args.gui_play_mode:
                gui = GameSimulatorGUI(root, 1, simulator, base_path)
            else:
                gui = GameSimulatorGUI(root, 2, simulator, base_path)

    win, lose, iter_cnt = 0, 0, 0

    def log_map():
        """
        현재 맵 상태를 로그로 출력합니다.
        """
        for x in range(len(simulator.map[0])):
            print_text = ""
            for y in range(len(simulator.map[0][x])):
                if simulator.map[0][x][y] == 0:
                    print_text += "□"
                elif simulator.map[0][x][y] == 1:
                    print_text += "■"
                elif simulator.map[0][x][y] == 2:
                    print_text += "X"
                elif simulator.map[0][x][y] == 3:
                    print_text += "X"
            print(print_text)
        if len(simulator.replay_text[0]) > 0:
            print(simulator.replay_text[0][-1] + "\n")
        print()

    def step_simulation():
        nonlocal recv_data, win, lose, iter_cnt
        # 로그로 맵 출력
        print(f"====== Turn {iter_cnt + 1} ======")
        log_map()

        # 시뮬레이터 데이터 업데이트
        # 듀얼 모드에서 플레이어가 조작할 경우
        if args.dual_mode and args.gui_play_mode:
            # Agent 선택 액션
            result = RandomAgent.select_action(recv_data, data_spec)

            # 플레이어 선택 액션
            player_result = return_box()
            player_result = gui.player_gui.simulate_play_mode_action(player_simulator, player_result)

            player_return_data_buffer = make_return_data('step', result_data=player_result, show_result=not args.skip_show_result)
            player_simulator.step(player_return_data_buffer['result_data'])
        else:
            # 기존 단일 모드 진행
            if not args.gui_play_mode:
                result = RandomAgent.select_action(recv_data, data_spec)
            else:
                result = return_box()
                result = gui.simulate_play_mode_action(simulator, result)

        return_data_buffer = make_return_data('step', result_data=result, show_result=not args.skip_show_result)
        recv_data = simulator.step(return_data_buffer['result_data'])

        # 결과 확인
        if recv_data['obs']['game_result'][0][2] == 0:
            if recv_data['obs']['game_result'][0][3] == 1:
                win += 1
            else:
                lose += 1
            print(f"Win: {win}, Lose: {lose}")

        # GUI 업데이트
        if args.dual_mode:
            # True 상태 업데이트
            gui.agent_gui.simulate_data_update(simulator, True)

            gui.agent_gui.simulate_data_update(simulator, False)
        elif args.use_gui and not args.gui_play_mode:
            # True 상태 업데이트
            gui.simulate_data_update(simulator, True)

            # 5초 후 False 상태 업데이트
            root.after(5000, lambda: gui.simulate_data_update(simulator, False))

        iter_cnt += 1

        # 종료 조건
        if iter_cnt < moves:
            if args.use_gui and not args.gui_play_mode:
                root.after(5000, step_simulation)
            else:
                step_simulation()
        else:
            print("Simulation Complete")

    # 시뮬레이션 시작
    if args.use_gui:
        step_simulation()
    else:
        # GUI 없이 시뮬레이션 실행
        for _ in range(moves):
            step_simulation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_show_result', action='store_true')
    parser.add_argument('--use_gui', action='store_true', default=True, help="Enable or disable GUI")
    parser.add_argument('--gui_play_mode', action='store_true', default=False, help="Enable or disable Play")
    parser.add_argument('--dual_mode', action='store_true', default=False, help="Enable or disable Dual Mode")

    args = parser.parse_args()
    args.skip_show_result = True

    # args.gui_play_mode = False
    if args.use_gui:
        root = tk.Tk()
        run_test(args, root)
        root.mainloop()
    else:
        args.gui_play_mode = False
        run_test(args)