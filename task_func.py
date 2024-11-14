from vehicle import ArmBase, ScreenShow, Key4Btn, ServoBus, ServoPwm
import cv2
import time
import numpy as np
import yaml, os

    
class MyTask:
    def __init__(self):
        # 旋转舵机
        self.servo_rotate = ServoBus(2)
        self.servo_rotate.set_angle(90, 0)
        
        # 放球舵机
        self.servo_ball = ServoBus(3)
        self.servo_ball.set_angle(80, -110)

        time.sleep(0.3)
        self.servo_ball.set_rotate(0)
        self.servo_rotate.set_rotate(0)
        self.servo_high = ServoPwm(2)

        self.block_cfg = {"arm": {"grap": [-0.2, 0.02], "release": [-0.2, 0.02]},
                          "target":[-0.2, 0.02]} 
        self.block_mid_pos = [-0.12, 0.04]

        # 机械臂
        self.angle_ball = [-75, -95]
        # self.cylinder_height = 0.14
        # 机械臂
        self.arm = ArmBase()

    def reset(self):
        self.arm.reset()
        self.arm.switch_side(-1)

    def put_down_self_block(self):
        target_height = 0.05
        target_horiz = -0.12
        arm_angle = 41
        grab_angle = 130

        # 平台达到指定角度，放平
        self.servo_ball.set_angle(80, -80)

        # 按照给定速度到达给定位置
        self.arm.set(target_horiz, target_height+0.03)
        # 到达指定角度
        self.arm.set_arm_angle(arm_angle)
        time.sleep(0.5)
        # 下移0.03m
        self.arm.set_offset(0, -0.02, 0.5)
        # 指定角度
        self.arm.set_grap_angle(grab_angle)
        time.sleep(0.4)
        # 上移0.05m
        self.arm.set_offset(0, 0.03, 0.8)
        # 返回初始角度
        self.arm.set_arm_dir(-1)
        time.sleep(0.5)
        self.arm.set_offset(0, -0.1, 1.5)

    def pick_up_block_self(self):
        target_height = 0.05
        target_horiz = -0.11
        arm_angle = 41
        grap_angle = 98
        
        # 到达指定位置上方
        self.arm.set(target_horiz, target_height+0.05)
        # 打开抓手,并到达指定角度
        self.arm.set_grap_angle(grap_angle+20)
        self.arm.set_arm_angle(arm_angle)
        time.sleep(0.5)
        # 下移0.03
        self.arm.set_offset(0, -0.03, 0.8)
        # 抓取物品
        self.arm.set_grap_angle(grap_angle)
        time.sleep(0.4)
        # 上移0.05
        self.arm.set_offset(0, 0.03, 0.8)
        # 抓手回到初始角度
        self.arm.set_arm_dir(-1)
        time.sleep(0.5)
        
    
    def elevation_pole(self, arm_set=False):
        pt_tar = []
        if arm_set:
            self.servo_rotate.set_angle(90, 0)
            time.sleep(0.3)
            self.servo_rotate.set_rotate(0)
            return pt_tar
        self.servo_rotate.set_rotate(100)
        time.sleep(5)
        self.servo_rotate.set_rotate(0)
        self.servo_rotate.set_angle(90, 0)
        time.sleep(0.3)
        self.servo_rotate.set_rotate(0)

    def pick_up_block(self, arm_set=False):
        # 定位目标的参数 label_id, obj_id, label, prob, err_x, err_y, width, height
        block_pt = [0, 0, "block", 0, -0.03, -0.28, 0.57, 0.8]
        # 转换到右侧
        self.arm.switch_side(-1)
        tar_horiz = -0.20
        tar_height = 0.03
        angle_grab_ball = 100
        
        # 抓手打开指定角度稍微大角度, 到达指定位置上方
        self.arm.set_grap_angle(angle_grab_ball+25)
        self.arm.set(tar_horiz+0.1, tar_height)
        if arm_set:
            
            
            return block_pt
        self.arm.set(tar_horiz+0.05, tar_height)
        # 右移0.05
        self.arm.set_offset(-0.05, 0, 0.6)
        # 抓取
        self.arm.set_grap_angle(angle_grab_ball)
        time.sleep(0.5)
        # 上升
        self.arm.set_offset(0, 0.02, 0.6)
        self.arm.set_offset(0.06, 0, 1)

    def put_down_block(self):
        tar_height = 0
        tar_horiz = -0.22
        angle_grab_ball = 135

        self.arm.set(tar_horiz+0.10, tar_height+0.05)
        # 右移0.1
        self.arm.set_offset(-0.13, 0, 1.5)
        # 下降
        self.arm.set_offset(0, -0.06, 0.8)
        # 松开
        self.arm.set_grap_angle(angle_grab_ball)
        time.sleep(0.5)
        # 上升
        self.arm.set_offset(0, 0.05, 0.8)
        # 左移0.1
        self.arm.set_offset(0.1, 0)
        
    def pick_up_ball(self, arm_set=False):
        # 定位目标的参数 label_id, obj_id, label, prob, err_x, err_y, width, height
        tar = [4, 0, 'blue_ball', 0.6, -0.05, 0.15, 0.47, 0.62]
        # 转换到左侧
        self.arm.switch_side(1)
        tar_height = 0.09
        angle_grab_ball = 95
        tar_horiz = -0.12
        self.arm.set(tar_horiz, tar_height)
        if arm_set:
            self.servo_ball.set_angle(80, -110)
            time.sleep(0.3)
            self.servo_ball.set_rotate(0)
            return tar
        
        self.arm.set_grap_angle(angle_grab_ball+25)
        self.arm.set_offset(0.06, 0, 0.6)
        self.arm.set_offset(0.05, 0, 0.6)
        self.arm.set_grap_angle(angle_grab_ball)
        time.sleep(0.5)
        self.arm.set_offset(0, 0.03, 0.2)
        self.arm.set_offset(-0.11, 0, 1)

    def put_down_self_ball(self):
        tar_height = 0.10
        tar_horiz = -0.12
        arm_angle = -52
        grap_angle = 130

        self.arm.set(tar_horiz, tar_height)
        self.arm.set_arm_angle(arm_angle)
        time.sleep(0.8)
        # self.arm.set(-0.1, 0.105, 0.3)
        self.arm.set_grap_angle(grap_angle)
        time.sleep(0.4)
        # self.arm.set(-0.1, 0.105, 0.3)
        self.arm.set_arm_dir(1)
        time.sleep(0.5)

    # 抓圆柱，选则大小
    def pick_up_cylinder(self, radius, arm_set=False):
        # 定位目标的参数 label_id, obj_id, label, prob, err_x, err_y, width, height
        tar_list =  [[1, 0, "cylinder1", 0,  -0.02, -0.22, 0.70, 0.7], [2, 0, "cylinder2", 0, -0.02, -0.22, 0.58, 0.69], 
                     [3,  0, "cylinder3", 0, -0.02, -0.23, 0.44, 0.69]]
        pt_tar = tar_list[radius-1]
        tar_height = 0.015
        #  99  126  148
        angle_list = [116, 110, 95]
        height_list = [0.08, 0.08, 0.14]
        angle_grab_ball = angle_list[radius-1]
        
        # dis_move = dis_list[radius-1]
        # print(angle_grab_ball)
        dis_offset = [0.01, 0.005, 0.0]
  
        # self.arm.reset_pos_dir(1, 0.05)
        self.arm.set_grap_angle(angle_grab_ball+25)
        # self.arm.set(tar_horiz - 0.12 * self.arm.side, tar_height)
        self.arm.set(-0.125, tar_height)
        if arm_set:
            return pt_tar
        horiz_offset = (0.11 - dis_offset[radius-1]) * self.arm.side
        self.arm.set_offset(horiz_offset, 0)
        # self.arm.set_offset(-0.12 + dis_offset, 0)
        # self.arm.set_offset(0.05 * self.arm.side, 0, 0.7)
        # self.arm.set_offset(0.07 *self.arm.side, 0, 1.1)
        self.arm.set_grap_angle(angle_grab_ball)
        time.sleep(0.5)
        self.arm.set_offset(0, height_list[radius-1])
        # self.arm.set_offset(0, 0.08, 1.3)

    def put_down_cylinder(self, radius):
        height_list = [0.07, 0.03, 0.03]
        tar_height = height_list[radius-1]
        angle_list = [123, 113, 95]
        angle_open = [25, 20, 20]
        angle_grab_ball = angle_list[radius-1] + angle_open[radius-1]

        dis_offset = [0.01, 0.005, 0.0]
        horiz_offset = (0.12 - dis_offset[radius-1]) * self.arm.side
        self.arm.set_offset(0, 0-tar_height, speed=[0.1, 0.05])
        time.sleep(0.2)
        self.arm.set_grap_angle(angle_grab_ball)
        time.sleep(0.5)
        self.arm.set_offset(0-horiz_offset, 0, 1.3)
    
    def put_down_ball(self):
        self.servo_ball.set_angle(60, -100)
        time.sleep(0.3)
        self.servo_ball.set_angle(80, -35)
        time.sleep(1.2)
        self.servo_ball.set_angle(80, -110)
        # time.sleep(0.6)
    
    def pick_high_ball(self, arm_set=False):
        tar_pos = [-0.12, 0.15]
        tar_height = 0.10
        tar_horiz = -0.12
        angle_high = 20 
        self.servo_high.set(80, angle_high)
        self.arm.set(tar_horiz, tar_height)
        # time.sleep(1)
        if arm_set:
            return
        self.arm.set_offset(0, 0.12)
        # self.arm.set_grap_angle(90)
        # time.sleep(1)
        self.servo_high.set(50, angle_high+32)
        time.sleep(1)
        self.arm.set_offset(0, -0.11)

    def task_test(self):
        self.servo_high.set(50, 50)

    def punish_crimall(self, arm_set=False):
        # 定位目标的参数 label_id, obj_id, label, prob, err_x, err_y, width, height
        tar = [0, 1, 'pedestrian',  0, -0.02, 0.4, 0.22, 0.82]
        tar_pos = [-0.12, 0.15]
        tar_height = 0.04
        tar_horiz = -0.12
        self.arm.switch_side(-1)
        self.arm.set(tar_horiz, tar_height)
        # time.sleep(1)
        if arm_set:
            return tar
        self.arm.set_grap_angle(80)
        self.arm.set_offset(-0.1, 0)
        self.arm.set_offset(0.1, 0)
        
    def ocr_arm_ready(self, side=-1):
        self.arm.switch_side(side)
        tar_height = 0.04
        tar_horiz = -0.12
        self.arm.set(tar_horiz, tar_height)

def task_reset():
    task = MyTask()
    task.reset()

def block_reset():
    task = MyTask()
    task.reset()
    task.pick_up_block()
    task.put_down_self_block()
    # task.pick_up_block_self()
    # task.put_down_block()

def ball_test():
    task = MyTask()
    
    # 抓三个球
    for i in range(3):
        task.pick_up_ball()
        task.put_down_self_ball()
    
    task.put_down_ball()

def cylinder_test():
    task = MyTask()
    i = 0
    tar = task.pick_up_cylinder(i+1, arm_set=True)
    time.sleep(0.8)
    task.pick_up_cylinder(i+1)
    time.sleep(0.5)
    task.put_down_cylinder(i+1)
    time.sleep(0.5)
    # for i in range(3):
    #     tar = task.pick_up_cylinder(i+1, arm_set=True)
    #     time.sleep(0.8)
    #     task.pick_up_cylinder(i+1)
    #     time.sleep(0.5)
    #     task.put_down_cylinder(i+1)
    #     time.sleep(0.5)

def highball_test():
    task = MyTask()
    task.pick_high_ball()

def punish_crimall_test():
    task = MyTask()
    task.punish_crimall()

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--op', type=str, default="reset")
    args = args.parse_args()
    print(args)
    if args.op == "reset":
        task_reset()
    if args.op == "stop":
        punish_crimall_test("infer_back_end.py")
    task_reset()
    # ball_test()
    # cylinder_test()
    # punish_crimall_test()
    # task = MyTask()
    # highball_test()
