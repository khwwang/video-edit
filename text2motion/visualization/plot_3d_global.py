import torch 
import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import imageio


def plot_3d_motion(args, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')
    
    
    joints, out_name, title = args
    
    data = joints.copy().reshape(len(joints), -1, 3)
    
    nb_joints = joints.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    limits = 1000 if nb_joints == 21 else 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    
    ########
    print(data.shape)
    #data[:, :, 1] -= np.mean(data[:, :, 1])
    
    def update(index):
        # 새로운 Figure와 Axes 객체 생성
        fig = plt.figure(figsize=(10, 10), dpi=96)
        ax = fig.add_subplot(111)  # 2D Axes 생성
        # 1. y축 기준 변화량 계산
        y_diff = np.abs(np.diff(data[:, :, 1], axis=0))  # y축 변화량 계산 (프레임 간)

# 2. 프레임별 변화량이 가장 큰 키포인트 찾기
        max_diff_keypoints = np.argmax(y_diff, axis=1)  # 각 프레임에서 y축 변화량이 가장 큰 키포인트 인덱스

# 3. 해당 키포인트를 0으로 설정
        for frame_idx, keypoint_idx in enumerate(max_diff_keypoints):
            data[frame_idx, keypoint_idx, :] = np.nan  # 프레임의 해당 키포인트 전체 좌표를 0으로 설정

        # Axes 초기화
        ax.clear()  # 이전 프레임 초기화

        # 5. 2D 관절 점들에 대한 x, y 값의 최소값과 최대값 계산
        min_x, max_x = np.nanmin(data[:, 0]), np.nanmax(data[:, 0])
        min_y, max_y = np.nanmin(data[:, 1]), np.nanmax(data[:, 1])
        range_x = max_x - min_x
        range_y = max_y - min_y
        ax.set_xlim(min_x - 0.1 * range_x, max_x + 0.1 * range_x)  # 여유를 두어 x축 범위 설정
        ax.set_ylim(min_y - 0.1 * range_y, max_y + 0.1 * range_y)  # 여유를 두어 y축 범위 설정
        
         # 중심을 기준으로 x, y축 범위 설정
        # range_x = MAXS[0] - MINS[0]
        # range_y = MAXS[1] - MINS[1]
        # ax.set_xlim(-range_x / 2, range_x / 2)  # x축 범위 설정
        # ax.set_ylim(-range_y / 2, range_y / 2)  # y축 범위 설정
        
        # 2D 관절 연결선 그리기
        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            linewidth = 4.0 if i < 5 else 2.0  # 주요 체인은 두껍게
            ax.plot(
                data[index, chain, 0],  # x 좌표
                data[index, chain, 1],  # y 좌표
                linewidth=linewidth,
                color=color
            )

        # 2D 관절 점 그리기
        ax.scatter(
            data[index, :, 0],  # x 좌표
            data[index, :, 1],  # y 좌표
            color='red',
            s=50
        )

        # 축 숨김
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.show()  # 디버깅 시 바로 출력

        # 이미지 저장 또는 반환
        if out_name is not None:
            plt.savefig(out_name, dpi=96)
            plt.close()
        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            arr = np.reshape(
                np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)
            )
            io_buf.close()
            plt.close()
            return arr



    # def update(index):
    #     # 새로운 Figure와 Axes3D 객체 생성
    #     fig = plt.figure(figsize=(10, 10), dpi=96)
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Axes 초기화
    #     ax.clear()  # 이전 프레임 초기화
    #     ax.set_xlim(MINS[0], MAXS[0])
    #     ax.set_ylim(MINS[1], MAXS[1])
    #     ax.set_zlim(MINS[2], MAXS[2])

    #     ax.grid(False)
    #     ax.view_init(elev=110, azim=-90)
    #     ax.dist = 7.5

    #     # 관절 연결선 그리기
    #     for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
    #         linewidth = 4.0 if i < 5 else 2.0
    #         ax.plot3D(
    #             data[index, chain, 0],
    #             data[index, chain, 1],
    #             data[index, chain, 2],
    #             linewidth=linewidth,
    #             color=color
    #         )
        
    #     # 관절 점 그리기 (항상 마지막에 실행)
    #     ax.scatter(
    #         data[index, :, 0],
    #         data[index, :, 1],
    #         data[index, :, 2],
    #         color='red',
    #         s=50
    #     )

    #     plt.axis('off')
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    #     ax.set_zticklabels([])

    #     plt.show()  # 디버깅 시 바로 출력

    #     # 이미지 저장 또는 반환
    #     if out_name is not None:
    #         plt.savefig(out_name, dpi=96)
    #         plt.close()
    #     else:
    #         io_buf = io.BytesIO()
    #         fig.savefig(io_buf, format='raw', dpi=96)
    #         io_buf.seek(0)
    #         arr = np.reshape(
    #             np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
    #             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)
    #         )
    #         io_buf.close()
    #         plt.close()
    #         return arr



    out = []
    for i in range(frame_number) : 
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)


def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None) : 
    
    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size) : 
        out.append(plot_3d_motion([smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None]))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), fps=20)
    out = torch.stack(out, axis=0)
    return out
    