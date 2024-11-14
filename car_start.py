#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import threading
import os
import numpy as np
from task_func import MyTask
from log_info import logger
from car_wrap import MyCar
from tools import CountRecord
import math
import sys, os
# 添加上本文件对应目录
sys.path.append(os.path.abspath(os.path.dirname(__file__))) 

if __name__ == "__main__":
    # kill_other_python()
    my_car = MyCar()
    my_car.STOP_PARAM = False
    # my_car.task.reset()
    
    def grap_block_func():
        # 设置手臂位置,返回抓取位置的信息
        pt_tar = my_car.task.pick_up_block(arm_set=True)
        time.sleep(1)
        # 设置手臂准备位置
        pt_tar = my_car.task.pick_up_block(arm_set=True)
        # 巡航8cm
        my_car.lane_dis_offset(0.3, 0.13)
        # 巡航右侧感应器感应到物块，并继续巡航2cm
        

        # return
        for i in range(2):
            # 调整机械臂的位置, 获取抓取物体的位置
            pt_tar = my_car.task.pick_up_block(arm_set=True)
            # 速度0.2m/s巡航0.03m一段距离
            my_car.lane_dis_offset(0.2, 0.03, stop=False)
            # 巡航到右侧有障碍
            my_car.lane_sensor(0.3, value_h=0.1, sides=-1, stop=False)
            
            # 根据抓取物体的位置定位目标
            res = my_car.lane_det_location(0.2, pt_tar, side=-1)
            if not res:
                continue
            # 抓取物体
            my_car.task.pick_up_block()
            if i==0:
                # 放下物体
                my_car.task.put_down_self_block()


    def release_block_func():
        # 调整机械臂的位置
        my_car.task.arm.switch_side(-1)
        # 速度0.3m/s巡航2.3m
        my_car.lane_dis_offset(0.3, 2.3, stop=False)
        # 速度0.3m/s巡航右侧感应器感应到0.4m到障碍
        my_car.lane_sensor(0.3, value_h=0.4, sides=-1, stop=False)
        # 速度0.3m/s巡航0.1m
        my_car.lane_dis_offset(0.2, 0.1, stop=True)

        # 调整位置,这个根据巡航效果调整
        my_car.set_pos_offset([0.16, 0.03, -0.3], 0.5)
        # 放下一个方块
        my_car.task.put_down_block()
        # 前移0.08m
        my_car.set_pos_offset([0.08, 0, 0])
        # 放第二个方块
        my_car.task.pick_up_block_self()
        my_car.task.put_down_block()

    def get_ball_func():
        my_car.task.arm.switch_side(1)
        # 调整机械手位置准备抓球，返回识别目标的位置
        pt = my_car.task.pick_up_ball(arm_set=True)
        my_car.lane_dis_offset(0.3, 0.7)
        my_car.lane_sensor(0.3, value_h=0.4, sides=1, stop=True)
        start_dis = my_car.get_dis_traveled()
        for i in range(3):
            # 根据给定目标和位置、方向定位调整车子的位置
            pt = my_car.task.pick_up_ball(arm_set=True)
            res = my_car.lane_det_location(0.2, pt, side=1)
            if res:
                my_car.task.pick_up_ball()
                my_car.task.put_down_self_ball()
            else:
                # 距离超过0.4m就跳出
                if my_car.get_dis_traveled() - start_dis < 0.40:
                    logger.info("dis out {}".format(i))
                else:
                    logger.info("can not find ball")
                    break
                continue


    def elevation_pole_func():
        # 巡航0.2m
        my_car.lane_dis_offset(0.3, 0.2)
        # 巡航到左侧感应器感应到障碍
        my_car.lane_sensor(0.2, value_h=0.2, sides=1)
        # 调整位置, 根据巡航效果调整
        my_car.set_pos_offset([0.09, 0.06, 0], 1)
        my_car.task.elevation_pole()
        my_car.set_pos_offset([0.0, 0.02, 0], 0.5)
        # time.sleep(5)

    def get_high_ball_func():
        # 调整位置准备抓球
        pt = my_car.task.pick_high_ball(arm_set=True)
        my_car.lane_dis_offset(0.3, 1)
        # 飞机停车坪移开位置
        pt = my_car.task.pick_high_ball(arm_set=True)
        my_car.lane_sensor(0.3, value_h=0.4, sides=1)
        # my_car.lane_advance(0.2, dis_offset=0.01, value_h=0.2, sides=1)
        my_car.set_pos_offset([-0.065, 0.08, 0], 1)
        # # 调整位置准备抓球
        my_car.task.pick_high_ball()
        my_car.set_pos_offset([0, -0.07, 0], 0.7)


    def pull_ball_func():
        my_car.lane_dis_offset(0.3, 1)
        my_car.lane_sensor(0.4, value_h=0.30, sides=1)
        # 调整位置准备放置球
        my_car.lane_dis_offset(0.21, 0.19)
        my_car.set_pos_offset([0, 0.05, 0], 0.7)
        my_car.task.put_down_ball()
 

    def hanoi_tower_func():
        my_car.lane_dis_offset(0.3, 0.9)
        det_side = my_car.lane_det_dis2pt(0.2, 1.2)
        side = my_car.get_card_side()

        # 调整检测方向
        my_car.task.arm.switch_side(side*-1)
        # 调整车子朝向
        my_car.set_pos_offset([0, 0, math.pi/4*1.5*side], 1)
        
        # 第一个要抓取的圆柱
        cylinder_id = 1
        # 调整抓手位置，获取要抓取的圆柱信息
        pt = my_car.task.pick_up_cylinder(cylinder_id, True)
        # 走一段距离
        my_car.lane_dis_offset(0.3,0.7)
        # 第二次感应到侧面位置
        my_car.lane_sensor(0.2, value_h=0.3, sides=side*-1)
        my_car.lane_sensor(0.2, value_l=0.3, sides=side*-1, stop=True)
        # 记录此时的位置
        pos_start = np.array(my_car.get_odom())
        logger.info("start pos:{}".format(pos_start))
        # my_car.lane_dis(0.2, 0.1)
        # return
        # 根据给定信息定位目标
        my_car.lane_det_location(0.2, pt, side=side*-1)
        # 抓取圆柱
        my_car.task.pick_up_cylinder(cylinder_id)
        # 计算走到记录位置的距离
        run_dis = my_car.calculation_dis(pos_start, np.array(my_car.get_odom()))
        # print("run_dis:{}".format(run_dis))
        # 后移刚才计算的距离，稍微多走一点儿
        my_car.set_pos_offset([0-(run_dis+0.065), 0, 0])
    
        # # print("stop pos:{}".format(my_car.get_odom()))
        tar_pos = my_car.get_odom()
        # 记录位置
        logger.info("tar_pos:{}".format(tar_pos))
        my_car.task.put_down_cylinder(cylinder_id)
        
        # 抓取2号圆柱
        cylinder_id = 2
        pt = my_car.task.pick_up_cylinder(cylinder_id, True)
        my_car.lane_det_location(0.2, pt, dis_out=0.5, side=-1*side)
        my_car.task.pick_up_cylinder(cylinder_id)
        my_car.set_pos(tar_pos)
        # print(my_car.get_odom())
        my_car.task.put_down_cylinder(cylinder_id)

        # 抓取3号圆柱
        cylinder_id = 3
        pt = my_car.task.pick_up_cylinder(cylinder_id, True)
        my_car.lane_dis_offset(0.2, 0.1)
        my_car.lane_det_location(0.2, pt, dis_out=0.5, side=-1*side)
        my_car.task.pick_up_cylinder(cylinder_id)
        my_car.set_pos(tar_pos)
        # print(my_car.get_odom())
        my_car.task.put_down_cylinder(cylinder_id)

        # 调整位置
        # my_car.task.pick_up_cylinder(cylinder_id, True)
        '''
        '''

    def camp_fun():
        # my_car.lane_advance(0.3, value_h=0.3, sides=-1)
        # time.sleep(25)
        my_car.lane_dis_offset(0.3, 1.5)
        # 调整位置准备进行ocr识别
        my_car.task.ocr_arm_ready(-1)
        # 感应到右侧障碍距离小于0.4
        my_car.lane_sensor(0.3, value_h=0.4, dis_offset=0.02, sides=-1, stop=True)
        text = my_car.get_ocr()
        if text is not None:
            
            logger.info("text:{}".format(text))
            actions_map = my_car.yiyan_get_actions(text)
            # 前移到营地左侧
            my_car.set_pos_offset([0.48, -0.11, -0.6], 2.5)
            pos_start = np.array(my_car.get_odom())
            # 离开道路到修整营地
            my_car.set_pos_offset([0.15, -0.4, 0], 2)
            my_car.delay(0.4)
            # 做任务
            my_car.do_action_list(actions_map)
            # 回到原来位置
            my_car.set_pos(pos_start)
        else:
            # 未检测到字，或者检测到的字不稳定，继续往下执行
            my_car.lane_dis_offset(0.3, 0.5)

    # 找到罪犯打击罪犯
    def find_criminal():
        my_car.lane_dis_offset(0.3, 0.5)
        # my_car.task.arm.switch_side(-1)
        # 调整位置准备进行ocr识别
        my_car.task.ocr_arm_ready(-1)
        # 感应到右侧障碍距离小于0.4
        my_car.lane_sensor(0.3, value_h=0.4, dis_offset=0.02, sides=-1, stop=True)
        text = my_car.get_ocr()
        if text is not None:
            logger.info("text:{}".format(text))
            
            criminal_attr = my_car.hum_analysis.get_res_json(text)
            # print(criminal_attr)
            # 调整机械手到识别的位置
            pt_tar = my_car.task.punish_crimall(arm_set=True)
            # 巡航到识别位置
            my_car.lane_sensor(0.1, value_h=0.4, sides=-1, stop=True)
            # my_car.location_det()
            for i in range(4):
                my_car.lane_det_location(0.1, pt_tar, det="mot", side=-1)
                attr_hum = my_car.get_hum_attr(pt_tar)
                res = my_car.compare_humattr(criminal_attr, attr_hum)
                if res:
                    logger.info("找到罪犯")
                    my_car.task.punish_crimall()
                    break
                else:
                    if i!=4:
                        # 前往下一个位置
                        my_car.set_pos_offset([0.07, 0, 0])
            else:
                logger.info("没有找到罪犯")
        else:
            # 未检测到字，或者检测到的字不稳定，继续往下执行
            my_car.lane_dis_offset(0.3, 0.4)
        # 返回起始地
        # my_car.lane_dis_offset(0.3, 0.5)

    def go_start():
        my_car.lane_sensor(0.3, value_l=0.4, sides=-1)
        my_car.set_pos_offset([0.85, 0, 0], 2.8)
        # 前移
        # my_car.set_pos_offset([0.3, 0, 0], 2.5)
        # my_car.set_pos_offset([0.45, -0.09, -0.6], 2.5)
        # 离开道路到修整营地
        # my_car.set_pos_offset([0.15, -0.4, 0], 2)
        # 做任务
        # my_car.do_action_list(actions_map)

    my_car.beep()
    time.sleep(0.2)
    functions = [grap_block_func, release_block_func, get_ball_func, elevation_pole_func, get_high_ball_func, 
                 pull_ball_func, hanoi_tower_func, camp_fun, find_criminal, go_start]
    my_car.manage(functions, 4)

