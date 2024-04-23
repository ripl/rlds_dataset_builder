import numpy as np
import tqdm
import os
import pickle
import cv2

N_TRAIN_EPISODES = 100
N_VAL_EPISODES = 100

EPISODE_LENGTH = 10

# load data from .pt file
def create_episode_leapmotion_format(file_path='0.pt', visualize=True):
    data = pickle.load(open(file_path, 'rb'))

    if visualize:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    episode = []
    for i in range(len(data)):
        state = np.concatenate([data[i]['gripper_pos'], data[i]['gripper_quat'], np.array([data[i]['cmd_grasp_pos']], dtype=np.float32)])
        action = np.concatenate([data[i]['cmd_trans_vel'], data[i]['cmd_rot_vel'], np.array([data[i]['cmd_grasp_pos']], dtype=np.float32)])
        image = data[i]['camarm']
        wrist_image = data[i]['camouter']

        if visualize:
            state_text = np.array2string(state, precision=2, separator=',', suppress_small=True)
            action_text = np.array2string(action, precision=2, separator=',', suppress_small=True)
            image_with_text = cv2.putText(np.concatenate([wrist_image, image], axis=1), f"State: {state_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            image_with_text = cv2.putText(image_with_text, f"Action: {action_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('image', image_with_text)
            cv2.waitKey(100)

        episode.append({
            'wrist_image': wrist_image,
            'image': image,
            'state': state,
            'action': action,
            'language_instruction': 'Put the yellow block on the rubik cube.',
        })

    if visualize:
        cv2.destroyAllWindows()

    np.save('data/train/episode_0.npy', episode)


if __name__ == '__main__':
    os.makedirs('data/val', exist_ok=True)
    os.makedirs('data/train', exist_ok=True)
    create_episode_leapmotion_format('0.pt', visualize=False)

    # # create fake episodes for train and validation
    # print("Generating train examples...")
    # os.makedirs('data/train', exist_ok=True)
    # for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
    #     create_fake_episode(f'data/train/episode_{i}.npy')

    # print("Generating val examples...")
    #
    # for i in tqdm.tqdm(range(N_VAL_EPISODES)):
    #     create_fake_episode(f'data/val/episode_{i}.npy')

    print('Successfully created example data!')
