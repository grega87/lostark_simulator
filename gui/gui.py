from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import random
import os

class DualGameSimulatorGUI:
    """
    플레이어와 AI의 두 개의 독립적인 GUI를 생성하고 관리하는 클래스.
    """
    def __init__(self, root, player_data, agent_data, base_path):
        self.root = root
        self.root.withdraw()  # 메인 윈도우 숨기기 (독립적인 창만 띄우기)

        # 플레이어 창 생성 (로그 없음)
        self.player_window = tk.Toplevel(root)
        self.player_gui = GameSimulatorGUI(self.player_window, mode=2, data=player_data, base_path=base_path, title="Player View", is_dual_mode=True)
        self.player_window.geometry("1280x720+0+0")  # x=100, y=100 위치

        # AI 창 생성 (로그 없음)
        self.agent_window = tk.Toplevel(root)
        self.agent_gui = GameSimulatorGUI(self.agent_window, mode=1, data=agent_data, base_path=base_path, title="Agent View", is_dual_mode=True)
        self.agent_window.geometry("960x720+960+0")  # x=1400, y=100 (오른쪽에 배치)

        # 플레이어가 턴을 마치면 AI의 턴이 자동으로 진행되도록 설정
        self.player_window.bind("<Return>", self.switch_turn)

    def switch_turn(self, event=None):
        """플레이어 턴 종료 후 AI 턴 실행"""
        print("플레이어 턴 종료, AI 턴 시작...")

        # AI가 자동으로 턴 진행
        self.agent_gui.simulate_data_change()

        # AI가 턴을 마친 후 다시 플레이어 턴으로 전환 (2초 후)
        self.player_window.after(2000, self.player_gui.update_field)

class GameSimulatorGUI:
    """
        게임 시뮬레이터의 그래픽 사용자 인터페이스를 관리하는 클래스입니다.
        시뮬레이션의 시각적 요소를 표시하고 모드에 따라 사용자 상호작용을 처리합니다.
    """
    def __init__(self, root, mode, data, base_path, title="LostArk_Simulator", is_dual_mode=False):
        """
               GameSimulatorGUI 클래스의 초기화 메서드입니다.

               매개변수:
               root -- Tkinter 루트 윈도우 객체
               mode -- 시뮬레이터 작동 모드 (예: '0 - default', '1 - agent', '2 - play')
               data -- 시뮬레이션 초기 데이터를 포함하는 딕셔너리
               base_path -- 리소스 파일들의 기본 디렉토리 경로

               GUI 요소들을 초기화하고 시뮬레이션에 필요한 초기 설정을 수행합니다.
       """
        self.main_path = base_path
        self.root = root
        self.root.title(title)
        self.mode = mode  # GUI 모드
        self.is_dual_mode = is_dual_mode

        # 듀얼 모드가 아닐 때만 로그 패널 생성
        if not self.is_dual_mode:
            self.log_frame = tk.Frame(self.root, width=300, bg="lightgray")
            self.log_frame.pack(side="left", fill="y")
            self.log_label = tk.Label(self.log_frame, text="Log", bg="lightgray", font=("Arial", 12, "bold"))
            self.log_label.pack(anchor="nw", padx=10, pady=5)
            self.log_text = tk.Text(self.log_frame, wrap="word", height=20, width=40)
            self.log_text.pack(padx=10, pady=5)
            self.log_text.config(state="disabled")

        if not mode == 0:
            if len(data.replay_text[0]) > 0:
                print(data.replay_text[0][-1])
            self.data = [{
                'common': [{
                    'Turn': data.turn[0],
                    'Info': 0,  # 실제 Info 데이터가 없어 0으로 설정
                    'curr_card1': data.card_list[0][1],
                    'curr_card2': data.card_list[0][0],
                    'next_card1': data.top_deck[0][0],
                    'next_card2': data.top_deck[0][1],
                    'next_card3': data.top_deck[0][2],
                    'reroll_number': data.reroll_list[0],
                    'left_turn': data.turn[0],
                    'card_level1': data.card_level_list[0][0],
                    'card_level2': data.card_level_list[0][1],
                }],

                'game_result': [{
                    'Result': 'Win' if data.game_result[0] == 1 else 'Lose',
                    'Score': data.game_result_array[0]
                }],

                'map': [{
                    'Location': f'Loc-{i}',
                    'Type': data.map[0][i // len(data.map[0])][i % len(data.map[0])]
                } for i in range(len(data.map[0]) * len(data.map[0]))],
                'step_result': [{
                    'Step': i,
                    'Outcome': 'Success' if data.step_result_array[0][i] > 0 else 'Failure'
                } for i in range(len(data.step_result_array[0]))],
                'make_position': [{
                    'Position': pos[0] * len(data.map[0]) + pos[1],
                    'Available': True
                } for pos in data.placeable_blocks[0]],

                'select_card': {f'card_{i}': 1 for i in range(len(data.card_list[0]))},

                'use_reroll': [{
                    'Roll': data.reroll_list[0],
                    'Used': data.is_apply_block_flag[0]
                }],
                'action': False  # 액션 상태 추가
            }]
        else:
            self.data = data  # 시뮬레이터 데이터
        self.images = []  # 이미지 참조를 유지하기 위한 리스트
        # Background 설정
        self.background_image = Image.open(os.path.join(self.main_path, "img", "map", "map.png"))
        self.background_image = self.background_image.resize((1280, 720))  # 배경 이미지 크기 조정
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        # 블록 이미지 로드
        self.block_photos = self.load_block_images()

        # 필드 생성
        self.canvas = tk.Canvas(self.root, width=1280, height=720)
        self.canvas.pack(fill="both", expand=True)

        # Background 설정
        self.canvas.create_image(0, 0, image=self.background_photo, anchor="nw")

        # 사용자 정의 중앙값 조정 변수
        self.offset_x = -25  # X축 중앙값 조정 (양수: 오른쪽 이동, 음수: 왼쪽 이동)
        self.offset_y = -80  # Y축 중앙값 조정 (양수: 아래 이동, 음수: 위 이동)

        self.done_select = False
        self.user_input_ready = False

        self.update_field()



    def update_log(self, message):
        """로그 메시지를 추가하는 메서드"""
        if not self.is_dual_mode:
            self.log_text.config(state="normal")  # 쓰기 가능 상태로 전환
            self.log_text.insert("end", message + "\n")  # 로그 추가
            self.log_text.see("end")  # 스크롤을 최신 로그로 이동
            self.log_text.config(state="disabled")  # 다시 읽기 전용으로 설정

    def load_block_images(self):
        # 블록 이미지 로드 (block01 ~ block09)
        block_photos = {}
        for i in range(10):
            img_path = os.path.join(self.main_path, "img", "map", f"block0{i}.png")
            if os.path.exists(img_path):
                block_image = Image.open(img_path).resize((30, 30))
                block_photos[i] = ImageTk.PhotoImage(block_image)
        return block_photos

    def update_field(self):
        # 맵 데이터 가져오기
        map_data = self.data[0]['map']
        map_width = int(len(map_data) ** 0.5)  # 맵의 너비를 동적으로 계산 (정사각형 가정)
        map_height = len(map_data) // map_width
        block_width, block_height = 30, 23  # 블록 크기


        # 화면 크기 가져오기
        canvas_width = 1280 + self.offset_x
        canvas_height = 720 + self.offset_y

        # 중앙 정렬 시작 좌표 계산
        start_x = (canvas_width - (map_width * block_width)) // 2
        start_y = (canvas_height - (map_height * block_height)) // 2


        self.canvas.delete("map")  # 기존 맵 삭제
        self.canvas.delete("map_select")  # 기존 맵 선택 삭제

        # 맵의 각 블록 표시
        for i in range(map_width):
            for j in range(map_height):
                map_index = i * map_height + j
                if map_index < len(map_data):
                    block = map_data[map_index]
                    x0 = start_x + i * block_width
                    y0 = start_y + j * block_height
                    block_type = block['Type']

                    # 블록 타입에 따른 이미지 표시
                    if block_type in self.block_photos:
                        self.canvas.create_image(
                            x0 + block_width // 2,
                            y0 + block_height // 2,
                            image=self.block_photos[block_type],
                            anchor="center",
                            tags="map"
                        )

                    # 선택된 위치 강조
                    if not self.data[0]['use_reroll'][0]['Used']:
                        if map_index == self.data[0]['make_position'][0]['Position']:
                            self.canvas.create_rectangle(
                                x0, y0, x0 + block_width, y0 + block_height,
                                outline="red", width=2, tags="map_select"
                            )

        # 카드 레벨 표기용
        card_level1 = self.data[0]['common'][0]['card_level1']
        card_level2 = self.data[0]['common'][0]['card_level2']

        # 카드 레벨에 따라 이펙트 이미지 추가
        card_level_effect_map = {
            2: "level2.png",
            3: "level3.png",
            # 필요하면 다른 레벨 추가
        }
        self.canvas.delete("card_level1_effect")
        self.canvas.delete("card_level2_effect")

        # 카드 레벨 1 이펙트
        if card_level1 in card_level_effect_map:
            level_img_path1 = os.path.join(self.main_path, "img", "cards", card_level_effect_map[card_level1])
            if os.path.exists(level_img_path1):
                level_img1 = Image.open(level_img_path1).resize((200, 290))  # 적절한 크기로 조정
                level_photo1 = ImageTk.PhotoImage(level_img1)

                self.images.append(level_photo1)  # 참조 유지
                self.canvas.create_image(510, 600, image=level_photo1, anchor="center", tags="card_level1_effect")

        # 카드 레벨 2 이펙트
        if card_level2 in card_level_effect_map:
            level_img_path2 = os.path.join(self.main_path, "img", "cards", card_level_effect_map[card_level2])
            if os.path.exists(level_img_path2):
                level_img2 = Image.open(level_img_path2).resize((200, 290))  # 적절한 크기로 조정
                level_photo2 = ImageTk.PhotoImage(level_img2)
                self.images.append(level_photo2)  # 참조 유지
                self.canvas.create_image(730, 600, image=level_photo2, anchor="center", tags="card_level2_effect")

        # 현재 패 카드 표시
        self.canvas.delete("current_cards")
        current_card_x = 510  # 중앙에 카드 배치
        curr_card1 = self.data[0]['common'][0]['curr_card1']
        curr_card2 = self.data[0]['common'][0]['curr_card2']
        current_cards = [curr_card1, curr_card2]

        for idx, card in enumerate(current_cards):
            img_path = os.path.join(self.main_path, "img", "cards", f"{card}.png")
            if os.path.exists(img_path):
                card_img = Image.open(img_path).resize((115, 160))
                card_photo = ImageTk.PhotoImage(card_img)
                self.images.append(card_photo)  # 참조 유지
                self.canvas.create_image(current_card_x, 680, image=card_photo, anchor="s",
                                         tags="current_cards")

                # 선택된 카드 외곽선 표시
                if self.data[0]['select_card'][f'card_{idx}'] == 1:
                    self.canvas.create_rectangle(
                        current_card_x - 60, 600 - 80, current_card_x + 60, 600 + 80,
                        outline="red", width=3, tags="current_cards"
                    )

                current_card_x += 220
            else:
                print(f"Card image not found for ID: {img_path}")
                current_directory = os.getcwd()
                print("현재 디렉토리:", current_directory)

        # 리롤 상태 처리
        self.canvas.delete("use_reroll")  # 이전 리롤 상태 초기화
        if self.data[0]['use_reroll'][0]['Used']:
            current_card_x = 510  # 중앙에 카드 배치
            for idx in range(2):  # 카드 개수만큼 반복
                if self.data[0]['use_reroll'][0]['Roll'] == idx + 1:
                    self.canvas.create_rectangle(
                        current_card_x - 60, 700 - 12, current_card_x + 60, 700 + 13,
                        outline="blue", width=3, tags="use_reroll"
                    )
                current_card_x += 220


        # 남은 턴 표시
        self.canvas.delete("turns_left")
        left_turn = self.data[0]['common'][0]['left_turn']
        self.canvas.create_text(
            532, 492, text=f"{left_turn}", font=("Arial", 10, "bold"), fill="yellow", tags="turns_left"
        )

        # 다음 정령 카드 표시
        self.canvas.delete("next_cards")
        self.canvas.create_text(200, 650, font=("Arial", 14), fill="white", tags="next_cards")
        next_card_x = 380
        next_card1 = self.data[0]['common'][0]['next_card1']
        next_card2 = self.data[0]['common'][0]['next_card2']
        next_card3 = self.data[0]['common'][0]['next_card3']
        next_cards = [next_card1, next_card2, next_card3]
        num = 1
        for card in next_cards:
            img_path = os.path.join(self.main_path, "img", "cards", f"{card}.png")

            if os.path.exists(img_path):
                if num == 1:
                    card_img = Image.open(img_path).resize((46, 70))
                else:
                    card_img = Image.open(img_path).resize((33, 50))
                card_photo = ImageTk.PhotoImage(card_img)
                self.images.append(card_photo)  # 참조 유지
                self.canvas.create_image(next_card_x, 680, image=card_photo, anchor="s", tags="next_cards")
                if num == 1:
                    next_card_x -= 70
                else:
                    next_card_x -= 60
                num = num + 1

        # 정령령 교체 가능 횟수
        reroll_number = self.data[0]['common'][0]['reroll_number']
        self.canvas.delete("reroll")
        self.canvas.create_text(620, 680, text=f"정령 교체", font=("Arial", 12, "bold"), fill="white", tags="reroll")
        self.canvas.create_text(620, 700, text=f"{reroll_number}회 가능", font=("Arial", 14, "bold"), fill="yellow", tags="reroll")

        # 초월 단계
        mode_text = "Player" if self.mode == 2 else "Agent" if self.mode == 1 else "Default"
        self.canvas.create_text(615, 25, text=f"초월 {mode_text} 진행중", font=("Arial", 12, "bold"), fill="yellow",
                                tags="reroll")

        # play 모드일 때만 이벤트 추가
        if self.mode == 2:
            # 선택 이벤트 바인딩
            self.canvas.bind("<Button-1>", self.handle_click)

        # 자동 업데이트 예제: 맵 상태 변경
        if self.mode == 0:
            self.root.after(2000, self.simulate_data_change)

    def simulate_data_change(self):
        # 데이터 변경 시뮬레이션
        for i in range(len(self.data[0]['map'])):
            self.data[0]['map'][i]['Type'] = random.randint(0, 8)  # 랜덤 블록 타입 설정

        # 현재 패와 다음 카드 시뮬레이션 변경
        self.data[0]['common'][0]['Turn'] = self.data[0]['common'][0]['Turn'] + 1
        self.data[0]['common'][0]['curr_card1'] = random.randint(0, 7)
        self.data[0]['common'][0]['curr_card2'] = random.randint(0, 7)
        self.data[0]['common'][0]['next_card1'] = random.randint(0, 7)
        self.data[0]['common'][0]['next_card2'] = random.randint(0, 7)
        self.data[0]['common'][0]['next_card3'] = random.randint(0, 7)
        self.data[0]['common'][0]['reroll_number'] = random.randint(0, 10)
        self.data[0]['common'][0]['left_turn'] = random.randint(0, 20)
        self.data[0]['select_card'] = {f'card_{i}': random.randint(0, 1) for i in range(2)}
        self.data[0]['use_reroll'] = [{'Roll': random.randint(0, 2), 'Used': random.choice([True, False])}]
        self.data[0]['make_position'][0]['Position'] = random.randint(0, 99)

        # 업데이트된 데이터로 화면 다시 그림
        self.update_field()

    def simulate_data_update(self, updata, action):
        # 필요한 값들을 변수로 선언
        turn = updata.turn[0]
        max_turn = updata.max_turn[0]
        curr_card1 = updata.card_list[0][0]
        curr_card2 = updata.card_list[0][1]
        next_card1, next_card2, next_card3 = updata.top_deck[0]
        reroll_number = updata.reroll_list[0]
        left_turn = turn
        game_result = 'Win' if updata.game_result[0] == 1 else 'Lose'
        game_score = updata.game_result_array[0]
        step_results = [
            {
                'Step': i,
                'Outcome': 'Success' if result > 0 else 'Failure'
            }
            for i, result in enumerate(updata.step_result_array[0])
        ]
        map_data = [
            {
                'Location': f'Loc-{i}',
                'Type': updata.map[0][i // len(updata.map[0])][i % len(updata.map[0])]
            }
            for i in range(len(updata.map[0]) * len(updata.map[0]))
        ]
        if action:
            self.data[0]['action'] = True
            # Print replay text if available
            if len(updata.replay_text[0]) > 0:
                # 전 선택 초기화
                self.data[0]['select_card'] = {f'card_{i}': 0 for i in range(len(updata.card_list[0]))}
                self.data[0]['make_position'] = []
                self.data[0]['use_reroll'][0]['Used'] = False  # 리롤 상태 초기화

                replay_text = updata.replay_text[0][-1]
                print(f'[GUI] : {replay_text}' + "\n")

                # Parse replay text
                if "use" in replay_text:  # 카드 사용
                    # Extract card index
                    parts = replay_text.split()
                    try:
                        card_index = int(parts[1])  # 카드 번호 (0: 왼쪽, 1: 오른쪽)
                    except (ValueError, IndexError):
                        print("Invalid card index in replay text.")
                        return  # 파싱 실패 시 함수 종료

                    # 좌표 추출 (문자열에서 "at" 이후의 좌표)
                    if "at" in replay_text:
                        coord_text = replay_text.split("at")[-1].strip()  # "at 0, 4" -> "0, 4"
                        x, y = map(int, coord_text.split(','))  # 좌표 파싱
                    else:
                        raise ValueError("No 'at' keyword found for position parsing.")

                    # Update selected card and position
                    self.data[0]['select_card'][f'card_{card_index}'] = 1  # 선택된 카드
                    self.data[0]['make_position'] = [{  # 선택된 위치 정보
                        'Position': x * len(updata.map[0]) + y,
                        'Card': updata.card_list[0][card_index],
                        'Action': 'Place'
                    }]
                    self.update_log(f"Card {card_index} used at position ({x}, {y}).")
                    print(f"[GUI] {self.data[0]['make_position']}")

                elif "reroll" in replay_text:  # 리롤
                    # Extract reroll index
                    parts = replay_text.split()
                    try:
                        reroll_index = int(parts[1])  # 리롤 번호 (1: 왼쪽, 2: 오른쪽)
                    except (ValueError, IndexError):
                        self.update_log("Invalid reroll index in replay text.")
                        return  # 파싱 실패 시 함수 종료

                    # Update reroll status
                    self.data[0]['use_reroll'][0]['Roll'] = reroll_index  # 리롤 사용
                    self.data[0]['use_reroll'][0]['Used'] = True  # 리롤 사용
                    self.update_log(f"Reroll performed on card {reroll_index}.")
        else:
            self.data[0]['action'] = False #액션 상태 추가
            # Initialize other game data
            self.data = [{
                'common': [{
                    'Turn': turn,
                    'Info': 0,  # 실제 Info 데이터가 없어 0으로 설정
                    'curr_card1': curr_card1,
                    'curr_card2': curr_card2,
                    'next_card1': next_card1,
                    'next_card2': next_card2,
                    'next_card3': next_card3,
                    'reroll_number': reroll_number,
                    'left_turn': left_turn,
                    'card_level1': updata.card_level_list[0][0],
                    'card_level2': updata.card_level_list[0][1],
                }],
                'game_result': [{
                    'Result': game_result,
                    'Score': game_score
                }],
                'map': map_data,
                'step_result': step_results,
                'make_position': self.data[0]['make_position'],  # 유지된 상태 반영
                'select_card': self.data[0]['select_card'],  # 유지된 상태 반영
                'use_reroll': self.data[0]['use_reroll']  # 유지된 상태 반영
            }]
            self.update_log(f'curr_card1 : {curr_card1}, curr_card2 : {curr_card2}')

        # 업데이트된 데이터로 화면 다시 그림
        self.update_field()

    def simulate_play_mode_action(self, simulator, return_data):
        """
        사용자 입력을 대기하고, 입력이 완료되면 주어진 return_box 객체를 수정합니다.
        모든 반환값은 ndarray 형식으로 반환됩니다.
        """
        # 1. 초기화 및 선택 화면 초기화
        self.select_position_reset()
        self.select_card_reset()
        self.select_reroll_reset()
        self.done_select = False
        self.user_input_ready = False

        # GUI에 시뮬레이터 상태 반영
        self.simulate_data_update(simulator, False)

        def on_user_input_complete():
            """
            사용자 입력이 완료되었을 때 호출.
            make_position과 select_card는 리스트 안에 ndarray 형식으로 반환.
            선택되지 않은 값은 0으로 반환.
            """
            # 데이터 추출 및 변환
            raw_make_position = self.data[0]['make_position'][0]['Position'] if self.data[0]['make_position'] else None
            make_position = (
                [np.array([raw_make_position], dtype=int)]
                if raw_make_position is not None
                else [np.array([0], dtype=int)]  # 선택되지 않았으면 기본값 0
            )

            # select_card를 리스트 안의 ndarray로 변환
            raw_select_card = [idx + 1 for idx, key in enumerate(self.data[0]['select_card']) if
                               self.data[0]['select_card'][key] == 1]
            select_card = (
                [np.array(raw_select_card, dtype=int)]
                if raw_select_card
                else [np.array([0], dtype=int)]  # 선택되지 않았으면 기본값 0
            )

            # use_reroll를 ndarray로 변환
            raw_use_reroll = [self.data[0]['use_reroll'][0]['Roll']] if self.data[0]['use_reroll'][0]['Used'] else []
            use_reroll = (
                np.array(raw_use_reroll, dtype=int)
                if raw_use_reroll
                else np.array([0], dtype=int)  # 선택되지 않았으면 기본값 0
            )

            # 선택 완료 여부 확인
            if self.done_select:
                # 선택된 데이터를 return_box 객체의 selected_action에 저장
                return_data.selected_action = {
                    'make_position': make_position,  # 리스트 안의 ndarray
                    'select_card': select_card,  # 리스트 안의 ndarray
                    'use_reroll': use_reroll  # ndarray
                }

                # 사용자 입력 완료 상태로 설정
                self.user_input_ready = True

        # GUI 이벤트 루프에서 사용자 입력 대기
        while not self.user_input_ready:
            self.root.update_idletasks()  # GUI 이벤트 처리
            self.root.update()  # GUI 화면 갱신
            on_user_input_complete()  # 사용자 입력 상태 확인

        return return_data

    # -------------- 실제 Play 모드 일때 핸들러 함수들 --------------
    def handle_click(self, event):
        """
        단일 클릭 이벤트 핸들러.
        클릭된 위치를 기준으로 리롤, 카드, 맵 클릭 이벤트를 처리.
        """
        if self.mode == 2:  # play 모드일 때만 동작
            x, y = event.x, event.y

            # 1. 리롤 선택 영역 확인
            self.handle_reroll_selection(event)

            # 2. 카드 선택 영역 확인
            self.handle_card_selection(event)

            # 3. 맵 클릭 영역 확인
            self.handle_position_selection(event)

    def select_position_reset(self):
        self.data[0]['make_position'] = [{'Position': None, 'Available': False}]
        self.canvas.delete("map_select")

    def select_card_reset(self):
        self.data[0]['select_card'] = {"card_0": 0, "card_1": 0}  # select_card 초기화
        self.canvas.delete("current_cards_select")

    def select_reroll_reset(self):
        self.data[0]['use_reroll'] = [{'Roll': None, 'Used': False}]  # use_reroll 초기화
        self.canvas.delete("use_reroll_select")

    # 리롤, 카드 선택, 위치 선택 부분에 클릭 이벤트 추가 (play 모드일 때만)
    def handle_card_selection(self, event):
        """
        카드 선택 이벤트 핸들러 (event 기반)
        두 개의 카드 중 하나만 선택 가능.
        """
        if self.mode == 2:  # play 모드일 때만 동작
            x, y = event.x, event.y
            current_card_x = 510  # 첫 번째 카드의 x 좌표
            card_width, card_height = 120, 160  # 카드 영역 크기

            for idx in range(2):  # 두 개의 카드 중 하나만 선택 가능
                x0 = current_card_x - card_width // 2
                x1 = current_card_x + card_width // 2
                y0 = 600 - card_height // 2
                y1 = 600 + card_height // 2

                if x0 <= x <= x1 and y0 <= y <= y1:
                    # 모든 카드 선택 초기화 (하나만 선택 가능)
                    self.data[0]['select_card'] = {f"card_0": 0, f"card_1": 0}

                    # 선택된 카드 업데이트 (1 또는 2로 설정)
                    self.data[0]['select_card'][f"card_{idx}"] = 1
                    self.canvas.delete("current_cards_select")
                    self.canvas.create_rectangle(
                        current_card_x - 60, 600 - 80, current_card_x + 60, 600 + 80,
                        outline="red", width=3, tags="current_cards_select"
                    )

                    # GUI에 선택 상태 반영
                    self.update_selection_display()

                    # 선택된 카드 번호 반환
                    return idx + 1  # 1 또는 2
                current_card_x += 220  # 다음 카드 위치로 이동

    def handle_position_selection(self, event):
        """
        위치 선택 이벤트 핸들러
        """
        # 선택된 카드 확인 (실제 카드 ID 가져오기)
        selected_card_indexes = [idx for idx, key in enumerate(self.data[0]['select_card']) if
                                 self.data[0]['select_card'][key] == 1]
        if not selected_card_indexes:
            print("카드를 먼저 선택해야 합니다.")
            return  # 카드를 선택하지 않으면 포지션 선택 불가

        # 실제 선택된 카드 ID 가져오기
        selected_card_ids = []
        if 0 in selected_card_indexes:
            selected_card_ids.append(self.data[0]['common'][0]['curr_card1'])  # 첫 번째 카드
        if 1 in selected_card_indexes:
            selected_card_ids.append(self.data[0]['common'][0]['curr_card2'])  # 두 번째 카드

        print(f"선택된 카드 ID: {selected_card_ids}")  # 디버깅용 출력

        if self.mode == 2:  # play 모드일 때만 동작
            x, y = event.x, event.y
            map_data = self.data[0]['map']
            map_width = int(len(map_data) ** 0.5)
            block_width, block_height = 30, 23
            start_x = (1280 + self.offset_x - (map_width * block_width)) // 2
            start_y = (720 + self.offset_y - (map_width * block_height)) // 2

            for i in range(map_width):
                for j in range(map_width):
                    map_index = i * map_width + j
                    if map_index < len(map_data):
                        x0 = start_x + i * block_width
                        y0 = start_y + j * block_height
                        block = map_data[map_index]

                        # 선택한 카드가 9번 또는 12번인 경우에만 맵 타입 3 선택 가능
                        if x0 <= x <= x0 + block_width and y0 <= y <= y0 + block_height:
                            if block['Type'] == 3 and not (9 in selected_card_ids or 12 in selected_card_ids):
                                print("맵 타입 3은 선택한 카드가 9번 또는 12번일 때만 선택 가능합니다.")
                                return
                            if block['Type'] == 0 or block == 2:
                                print("빈 블록은 선택 불가")
                                return
                            # 선택된 위치 업데이트
                            position = i * 8 + j
                            self.data[0]['make_position'] = [{'Position': position, 'Available': True}]
                            self.canvas.delete("map_select")
                            self.canvas.create_rectangle(
                                x0, y0, x0 + block_width, y0 + block_height,
                                outline="red", width=2, tags="map_select"
                            )
                            self.done_select = True
                            self.update_selection_display()
                            return

    def handle_reroll_selection(self, event):
        """
        리롤 선택 이벤트 핸들러 (event 기반)
        """
        if self.data[0]['common'][0]['reroll_number'] == 0:
            print("리롤 횟수 부족")
            return
        if self.mode == 2:  # play 모드일 때만 동작
            x, y = event.x, event.y
            current_card_x = 510  # 첫 번째 리롤 버튼의 x 좌표
            reroll_width, reroll_height = 120, 40  # 리롤 버튼 크기

            for idx in range(2):  # 두 개의 리롤 버튼
                x0 = current_card_x - reroll_width // 2
                x1 = current_card_x + reroll_width // 2
                y0 = 700 - reroll_height // 2
                y1 = 700 + reroll_height // 2

                if x0 <= x <= x1 and y0 <= y <= y1:
                    # 리롤 상태 업데이트
                    self.data[0]['use_reroll'][0]['Used'] = True
                    self.data[0]['use_reroll'][0]['Roll'] = idx + 1
                    self.canvas.delete("use_reroll_select")
                    self.canvas.create_rectangle(
                        current_card_x - 60, 700 - 12, current_card_x + 60, 700 + 13,
                        outline="blue", width=3, tags="use_reroll_select"
                    )
                    self.select_position_reset()
                    self.select_card_reset()
                    self.update_selection_display()
                    self.done_select = True
                    return
                current_card_x += 220  # 다음 리롤 버튼 위치로 이동

    def update_pred_display(self, pred_clear, save_game_turn):
        self.canvas.delete("pred_display")

        self.canvas.create_text(
            150, 80,
            text=f"클리어 확률 : {pred_clear:.2f}%",
            font=("Arial", 20, 'bold'),
            fill="white",
            tags="pred_display"
        )

        # self.canvas.create_text(
        #     150, 120,
        #     text=f"남은 턴수 예측 : {self.data[0]['common'][0]['left_turn'] - save_game_turn} 턴",
        #     font=("Arial", 20, 'bold'),
        #     fill="white",
        #     tags="pred_display"
        # )

    def update_selection_display(self):
        """
        현재 선택된 데이터를 GUI에 표시합니다.
        """
        self.canvas.delete("selection_display")  # 기존 표시 제거

        # 선택된 위치
        make_position = self.data[0]['make_position'][0]['Position'] if self.data[0]['make_position'] else "None"

        # 선택된 카드
        select_card = [key for key, value in self.data[0]['select_card'].items() if value == 1]
        select_card_text = select_card if select_card else "None"

        # 리롤 상태
        use_reroll = self.data[0]['use_reroll'][0]['Roll'] if self.data[0]['use_reroll'][0]['Used'] else "None"


    def game_result(self, result_text):
        # 결과 텍스트 (승패 정보)
        self.canvas.create_text(
            640, 150,  # 중앙 아래쪽
            text=result_text,
            font=("Arial", 30, "bold"),
            fill="yellow",
            tags="result_display"
        )

        # 중앙에 큰 글씨로 클릭 안내 메시지
        self.canvas.create_text(
            640, 200,  # 중앙 좌표
            text="클릭하여 진행",
            font=("Arial", 30, "bold"),
            fill="red",
            tags="result_display",
            anchor="center"
        )

        # 클릭 이벤트 감지 및 무한 대기
        clicked = [False]  # 리스트로 상태 저장 (mutable 객체 필요)

        def exit_loop(event):
            clicked[0] = True  # 클릭 상태 변경

        self.canvas.bind("<Button-1>", exit_loop)

        while not clicked[0]:  # 클릭될 때까지 무한 루프
            self.root.update_idletasks()
            self.root.update()

        self.canvas.delete("result_display")

if __name__ == "__main__":
    # 예제 데이터 설정
    simulator_number = 1
    data = [
        {
            'common': [{
                'Turn': 1,
                'curr_card1': random.randint(0, 7),
                'curr_card2': random.randint(0, 7),
                'next_card1': random.randint(0, 7),
                'next_card2': random.randint(0, 7),
                'next_card3': random.randint(0, 7),
                'reroll_number': random.randint(0, 10),
                'left_turn': random.randint(0, 20),
                'card_level1': 2,
                'card_level2': 3,
            }],
            'game_result': [{'Result': random.choice(['Win', 'Lose']), 'Score': random.randint(0, 100)}],
            'map': [{'Location': f'Loc-{i}', 'Type': random.randint(0, 8)} for i in range(100)],
            'step_result': [{'Step': 0, 'Outcome': random.choice(['Success', 'Failure'])}],
            'make_position': [{'Position': random.randint(0, 99), 'Available': True}],
            'select_card': {f'card_{i}': random.randint(0, 1) for i in range(2)},
            'use_reroll': [{'Roll': random.randint(0, 2), 'Used': random.choice([True, False])}],
            'action': False  # 액션 상태 추가
        }
        for _ in range(simulator_number)
    ]

    gui_mode = True  # GUI 모드 활성화 여부
    if gui_mode:
        root = tk.Tk()
        app = GameSimulatorGUI(root, 0, data, "")
        root.mainloop()
