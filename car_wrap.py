#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import threading
import os
import platform
import signal
from camera import Camera
import numpy as np
from vehicle import MecanumBase, SensorAi, SerialWrap, ArmBase, ScreenShow, Key4Btn, Infrared, LedLight
from simple_pid import PID
import cv2, math
from task_func import MyTask
from infer_cs import ClintInterface, Bbox
from ernie_bot import ErnieBotWrap, ActionPrompt, HumAttrPrompt
from tools import CountRecord, get_yaml, IndexWrap
import sys, os

# 添加上本地目录
sys.path.append(os.path.abspath(os.path.dirname(__file__))) 
from log_info import logger

def sellect_program(programs, order, win_order):
    dis_str = ''
    start_index = 0
    
    start_index = order - win_order
    for i, program in enumerate(programs):
        if i < start_index:
            continue

        now = str(program)
        if i == order:
            now = '>>> ' + now
        else:
            now = str(i+1) + '.' + now
        if len(now) >= 19:
            now = now[:19]
        else:
            now = now + '\n'
        dis_str += now
        if i-start_index == 4:
            break
    return dis_str

def kill_other_python():
    import psutil
    pid_me = os.getpid()
    # logger.info("my pid ", pid_me, type(pid_me))
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower() and len(proc.info['cmdline']) > 1 and len(proc.info['cmdline'][1]) < 30:
                    python_processes.append(proc.info)
            # 出现异常的时候捕获 不存在的异常，权限不足的异常， 僵尸进程
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    for process in python_processes:
        # logger.info(f"PID: {process['pid']}, Name: {process['name']}, Cmdline: {process['cmdline']}")
        # logger.info("this", process['pid'], type(process['pid']))
        if int(process['pid']) != pid_me:
            os.kill(int(process['pid']), signal.SIGKILL)
            time.sleep(0.3)
            
def limit(value, value_range):
    return max(min(value, value_range), 0-value_range)

# 两个pid集合成一个
class PidCal2():
    def __init__(self, cfg_pid_y=None, cfg_pid_angle=None):
        self.pid_y = PID(**cfg_pid_y)
        self.pid_angle = PID(**cfg_pid_angle)
    
    def get_out(self, error_y, error_angle):
        pid_y_out = self.pid_y(error_y)
        pid_angle_out = self.pid_angle(error_angle)
        return pid_y_out, pid_angle_out

class LanePidCal():
    def __init__(self, cfg_pid_y=None, cfg_pid_angle=None):
        # y_out_limit = 0.7
        # self.pid_y = PID(5, 0, 0)
        # self.pid_y.setpoint = 0
        # self.pid_y.output_limits = (-y_out_limit, y_out_limit)
        print(cfg_pid_y)
        print(cfg_pid_angle)
        self.pid_y = PID(**cfg_pid_y)
        print(self.pid_y)

        angle_out_limit = 1.5
        self.pid_angle = PID(3, 0, 0)
        self.pid_angle.setpoint = 0
        self.pid_angle.output_limits = (-angle_out_limit, angle_out_limit)
    
    def get_out(self, error_y, error_angle):
        pid_y_out = self.pid_y(error_y)
        pid_angle_out = self.pid_angle(error_angle)
        return pid_y_out, pid_angle_out
    
class DetPidCal():
    def __init__(self, cfg_pid_y=None, cfg_pid_angle=None):
        y_out_limit = 0.7
        self.pid_y = PID(0.3, 0, 0)
        self.pid_y.setpoint = 0
        self.pid_y.output_limits = (-y_out_limit, y_out_limit)

        angle_out_limit = 1.5
        self.pid_angle = PID(2, 0, 0)
        self.pid_angle.setpoint = 0
        self.pid_angle.output_limits = (-angle_out_limit, angle_out_limit)
    
    def get_out(self, error_y, error_angle):
        pid_y_out = self.pid_y(error_y)
        pid_angle_out = self.pid_angle(error_angle)
        return pid_y_out, pid_angle_out
    

class LocatePidCal():
    def __init__(self):
        y_out_limit = 0.3
        self.pid_y = PID(0.5, 0, 0)
        self.pid_y.setpoint = 0
        self.pid_y.output_limits = (-y_out_limit, y_out_limit)

        x_out_limit = 0.3
        self.pid_x = PID(0.5, 0, 0)
        self.pid_x.setpoint = 0
        self.pid_x.output_limits = (-x_out_limit, x_out_limit)
    
    def set_target(self, x, y):
        self.pid_y.setpoint = y
        self.pid_x.setpoint = x

    def get_out(self, error_x, error_y):
        pid_y_out = self.pid_y(error_y)
        pid_x_out = self.pid_x(error_x)
        return pid_x_out, pid_y_out

class MyCar(MecanumBase):
    STOP_PARAM = True
    def __init__(self):
        # 调用继承的初始化
        start_time = time.time()
        super(MyCar, self).__init__()
        logger.info("my car init ok {}".format(time.time() - start_time))
        # 任务
        self.task = MyTask()
        # 显示
        self.display = ScreenShow()
        
        # 获取自己文件所在的目录路径
        self.path_dir = os.path.abspath(os.path.dirname(__file__))
        self.yaml_path = os.path.join(self.path_dir, "config_car.yml")
        # 获取配置
        cfg = get_yaml(self.yaml_path)
        # 根据配置设置sensor
        self.sensor_init(cfg)

        self.car_pid_init(cfg)
        
        self.camera_init(cfg)
        # paddle推理初始化
        self.paddle_infer_init()
        # 文心一言分析初始化
        self.ernie_bot_init()

        # 相关临时变量设置
        # 程序结束标志
        self._stop_flag = False
        # 按键线程结束标志
        self._end_flag = False
        self.thread_key = threading.Thread(target=self.key_thread_func)
        self.thread_key.setDaemon(True)
        self.thread_key.start()
    
    def sensor_init(self, cfg):
        cfg_sensor = cfg['io']
        # print(cfg_sensor)
        self.key = Key4Btn(cfg_sensor['key'])
        self.light = LedLight(cfg_sensor['light'])
        self.left_sensor = Infrared(cfg_sensor['left_sensor'])
        self.right_sensor = Infrared(cfg_sensor['right_sensor'])
    
    def car_pid_init(self, cfg):
        # lane_pid_cfg = cfg['lane_pid']
        # self.pid_y = PID(lane_pid_cfg['y'], 0, 0)
        # self.lane_pid = LanePidCal(**cfg['lane_pid'])
        # self.det_pid = DetPidCal(**cfg['det_pid'])
        self.lane_pid = PidCal2(**cfg['lane_pid'])
        self.det_pid = PidCal2(**cfg['det_pid'])

    def camera_init(self, cfg):
        # 初始化前后摄像头设置
        self.cap_front = Camera(cfg['camera']['front'])
        # 侧面摄像头
        self.cap_side = Camera(cfg['camera']['side'])

    def paddle_infer_init(self):
        self.crusie = ClintInterface('lane')
        # 前置左右方向识别
        self.front_det = ClintInterface('front')
        # 任务识别
        self.task_det = ClintInterface('task')
        # 人体跟踪
        self.mot_hum = ClintInterface('mot')
        # 人体属性
        self.attr_hum = ClintInterface('humattr')
        # ocr识别
        self.ocr_rec = ClintInterface('ocr')
        # 识别为None
        self.last_det = None

    def ernie_bot_init(self):
        self.hum_analysis = ErnieBotWrap()
        self.hum_analysis.set_promt(str(HumAttrPrompt()))

        self.action_bot = ErnieBotWrap()
        self.action_bot.set_promt(str(ActionPrompt()))

    @staticmethod
    def get_cfg(path):
        from yaml import load, Loader
        # 把配置文件读取到内存
        with open(path, 'r') as stream:
            yaml_dict = load(stream, Loader=Loader)
        port_list = yaml_dict['port_io']
        # 转化为int
        for port in port_list:
            port['port'] = int(port['port'])
        print(yaml_dict)

    # 延时函数
    def delay(self, time_hold):
        start_time = time.time()
        while True:
            if self._stop_flag:
                return
            if time.time() - start_time > time_hold:
                break
            
    # 按键检测线程
    def key_thread_func(self):
        while True:
            if not self._stop_flag:
                if self._end_flag:
                    return
                key_val = self.key.get_btn()
                # print(key_val)
                if key_val == 3:
                    self._stop_flag = True
                time.sleep(0.2)
    
    
    # 根据某个值获取列表中匹配的结果
    @staticmethod
    def get_list_by_val(list, index, val):
        for det in list:
            if det[index] == val:
                return det
        return None
    
    def do_action_list(self, acitons):
        def move_func(x, y, angle):
            self.set_pos_offset([x, y, angle])

        def illuminate_func(time_dur):
            for i in range(1, 5):
                self.light.set_light(i, 255, 255, 255)
            self.delay(time_dur)
            for i in range(1, 5):
                self.light.set_light(i, 0, 0, 0)
        
        def blink_func(blink_count):
            for i in range(blink_count):
                for i in range(1, 5):
                    self.light.set_light(i, 255, 255, 255)
                self.delay(0.5)
                for i in range(1, 5):
                    self.light.set_light(i, 0, 0, 0)

        def beep_func(time_dur, count):
            for i in range(count):
                self.beep(sec = time_dur)

        def wait_func(wait_time):
            self.delay(wait_time)

        action_map = {
            'move': move_func,         # 移动功能
            'illuminate': illuminate_func,  # 持续照亮
            'blink': blink_func,       # 闪烁功能
            'beep': beep_func,         # 蜂鸣器发声
            'wait': wait_func          # 等待功能
        }

        for action in acitons:
            func = action_map[action['func']]
            # 删除func, 剩下的都是params
            action.pop('func')
            # 执行
            func(**action)

    # 计算两个坐标的距离
    def calculation_dis(self, pos_dst, pos_src):
        return math.sqrt((pos_dst[0] - pos_src[0])**2 + (pos_dst[1] - pos_src[1])**2)
    
    # 侧面摄像头进行位置定位
    def lane_det_location(self, speed, pt_tar=[0, 1, 'pedestrian',  0, -0.15, -0.48, 0.24, 0.82], dis_out=0.22, side=1, det='task'):
        
        infer = self.task_det
        if det!="task":
            infer = self.mot_hum
        pid_x = PID(0.5, 0, 0.02, setpoint=0, output_limits=(-speed, speed))
        pid_y = PID(1.3, 0, 0.01, setpoint=0, output_limits=(-0.15, 0.15))
        pid_w = PID(1.0, 0, 0.02, setpoint=0, output_limits=(-0.15, 0.15))

        # 用于相同记录结果的计数类
        x_count = CountRecord(10)
        y_count = CountRecord(10)
        w_count = CountRecord(10)
        
        out_x = speed
        out_y = 0
        # 坐标位置error转换相对位置
        error_adjust = np.array([-1, 1, 1, 1])
        if side == -1:
            error_adjust = np.array([1, -1, -1, -1])
        
        # 此时设置相对初始位置
        self.set_pos_relative()
        find_tar = False
        # 类别, id, 置信度, 归一化bbox[x_c, y_c, w, h]
        tar_cls, tar_id, tar_label, tar_score, tar_bbox = pt_tar[0], pt_tar[1], pt_tar[2], pt_tar[3], pt_tar[4:]
        flag_location = False
        while True:
            if self._stop_flag:
                return
            _pos_x, _pos_y, _pos_omage = self.get_pos_relative() # 用来计算距离

            if abs(_pos_x) > dis_out or abs(_pos_y) > dis_out:
                if not find_tar:
                    logger.info("task location dis out")
                    break
            img_side = self.cap_side.read()
            dets_ret = infer(img_side)
            # dets_ret = self.mot_hum(img_side)
            # cv2.imshow("side", img_side)
            # cv2.waitKey(1)
            # 进行排序，此处排列按照由近及远的顺序
            dets_ret.sort(key=lambda x: (x[4]-tar_bbox[0])**2 + (x[5]-tar_bbox[1])**2)
            # print(dets_ret)
            # 找到最近对应的类别，类别存在第一个位置
            det = self.get_list_by_val(dets_ret, 2, tar_label)
            # 如果没有，就重新获取
            if det is not None:
                find_tar = True
                # 结果分解
                det_cls, det_id, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
                # 计算偏差, 并进行偏差转换为输入pid的输入值
                bbox_error = ((np.array(det_bbox) - np.array(tar_bbox)) * error_adjust).tolist()
                # 离得远时. ywh值进行滤波为0，最终仅使用了w的值
                if abs(bbox_error[0]) > 0.1:
                    bbox_error[1] = 0
                    bbox_error[2] = 0
                    bbox_error[3] = 0
                out_x = pid_x(bbox_error[0])
                # out_y = pid_y(bbox_error[1])
                out_y = pid_w(bbox_error[2])
                # 检测偏差值连续小于阈值时，跳出循环
                # print(bbox_error)
                flag_x = x_count(abs(bbox_error[0]) < 0.02)
                
                flag_y = y_count(abs(bbox_error[2]) < 0.025)
                if flag_x:
                    out_x = 0
                if flag_y:
                    out_y = 0
                if flag_x and flag_y:
                    print("location ok")
                    flag_location = True
                    break
                
                # print("error_x:{:.2}, error_y:{:.2}, out_x:{:.2}, out_y:{:2}".format(bbox_error[0], bbox_error[2], out_x, out_y))
            else:
                x_count(False)
                y_count(False)
            self.mecanum_wheel(out_x, out_y, 0)
        # 停止
        self.mecanum_wheel(0, 0, 0)
        return flag_location
            
    def lane_det_base(self, speed, end_fuction, stop=STOP_PARAM):
        y_speed = 0
        angle_speed = 0
        while True:
            image = self.cap_front.read()
            dets_ret = self.front_det(image)
            # 此处检测简单不需要排序
            # dets_ret.sort(key=lambda x: x[4]**2 + (x[5])**2)
            if len(dets_ret)>0:
                det = dets_ret[0]
                det_cls, det_id, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
                error_y = det_bbox[0]
                dis_x = 1 - det_bbox[1]
                if end_fuction(dis_x):
                    break
                error_angle = error_y/dis_x
                y_speed, angle_speed = self.det_pid.get_out(error_y, error_angle)
            self.mecanum_wheel(speed, y_speed, angle_speed)
            if end_fuction(0):
                break
        if stop:
            self.stop()

    # 正面摄像头进行位置定位
    def lane_det_dis2pt(self, speed, dis_end, stop=STOP_PARAM):
        # lambda定义endfunction
        end_fuction = lambda x: x < dis_end and x != 0
        self.lane_det_base(speed, end_fuction, stop=stop)
        
    def lane_base(self, speed, end_fuction, stop=STOP_PARAM):
        while True:
            if self._stop_flag:
                return
            image = self.cap_front.read()
            error_y, error_angle = self.crusie(image)
            y_speed, angle_speed = self.lane_pid.get_out(-error_y, -error_angle)
            # speed_dy, angle_speed = process(image)
            self.mecanum_wheel(speed, y_speed, angle_speed)
            if end_fuction():
                break
        if stop:
            self.stop()

    
    # 巡航一段路程 绝对路程
    def lane_dis(self, speed, dis_end, stop=STOP_PARAM):
        # lambda重新endfunction
        end_fuction = lambda: self.get_dis_traveled() > dis_end
        self.lane_base(speed, end_fuction, stop=stop)

    # 巡航一段路程 相对路程
    def lane_dis_offset(self, speed, dis_hold, stop=STOP_PARAM):
        dis_start = self.get_dis_traveled()
        dis_stop = dis_start + dis_hold
        self.lane_dis(speed, dis_stop, stop=stop)

    # 巡航至检测到障碍
    def lane_sensor(self, speed, value_h=None, value_l=None, dis_offset=0.0, times=1, sides=1, stop=STOP_PARAM):
        if value_h is None:
            value_h = 1200
        if value_l is None:
            value_l = 0
        _sensor_usr = self.left_sensor
        if sides == -1:
            _sensor_usr = self.right_sensor
        # 用于检测开始过渡部分的标记
        flag_start = False
        def end_fuction():
            nonlocal flag_start
            val_sensor = _sensor_usr.get() / 1000
            # print("val:", val_sensor)
            if val_sensor < value_h and val_sensor > value_l:
                return flag_start
            else:
                flag_start = True
                return False

        for i in range(times):
            self.lane_base(speed, end_fuction, stop=False)
        # 根据需要是否巡航
        self.lane_dis_offset(speed, dis_offset, stop=stop)

    def get_card_side(self):
        # 检测卡片左右指示
        count_side = CountRecord(3)
        while True:
            if self._stop_flag:
                return
            image = self.cap_front.read()
            dets_ret = self.front_det(image)
            if len(dets_ret) == 0:
                count_side(-1)
                continue
            det = dets_ret[0]
            det_cls, det_id, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
            # 联系检测超过3次
            if count_side(det_label):
                if det_label == 'turn_right':
                    return -1
                elif det_label == 'turn_left':
                    return 1
    
    def get_hum_attr(self, pt_tar, show=False):
        # 类别, id, 置信度, 归一化bbox[x_c, y_c, w, h]
        tar_cls, tar_id, tar_label, tar_score, tar_bbox = pt_tar[0], pt_tar[1], pt_tar[2], pt_tar[3], pt_tar[4:]
        tar_count = CountRecord(4)
        while True:
            if self._stop_flag:
                return
            image = self.cap_side.read()
            dets_ret = self.mot_hum(image)
            # 排序, 按照距离的远近
            dets_ret.sort(key=lambda x: (x[4]-tar_bbox[0])**2 + (x[5]-tar_bbox[1])**2)
            if len(dets_ret) != 0:
                det_rect_normalise = dets_ret[0][4:]
                # print(det_rect_normalise)
                rect = Bbox(box=det_rect_normalise, size=image.shape[:2][::-1]).get_rect()
                # print(rect)
                if show:
                    cv2.rectangle(image, rect[:2], rect[2:], (0, 255, 0), 2)
                image_hum = image[rect[1]:rect[3], rect[0]:rect[2]]
                # cv2.imshow("hum", image_hum)
                # cv2.waitKey(1)
                # image_hum = image[int(det_rect[1]):int(det_rect[3]), int(det_rect[0]):int(det_rect[2])]
                res = self.attr_hum(image_hum)
                print(res)
                if tar_count(res):
                    # print(res)
                    return res
            if show:
                cv2.imshow("hum", image)
                cv2.waitKey(1)
            # print(res)
            # image_hum
            # logger.info(res)
                
    def compare_humattr(self, crimall_attr, hum_attr):
        if crimall_attr is None:
            return False
        for key, val in crimall_attr.items():
            if key not in hum_attr:
                return False
            if type(val) is bool:
                if val != hum_attr[key]:
                    return False
            elif val.lower() != hum_attr[key].lower():
                return False
        return True

    def get_ocr(self, time_out=10):
        time_stop = time.time() + time_out
        # 简单滤波,三次检测到相同的值，认为稳定并返回
        text_count = CountRecord(3)
        while True:
            if self._stop_flag:
                return
            if time.time() > time_stop:
                return None
            img = self.cap_side.read()
            text = self.ocr_rec(img[180:, 200:440])
            # print(text)
            if text_count(text):
                return text
            
    def yiyan_get_humattr(self, text):
        return self.hum_analysis.get_res_json(text)
    
    def yiyan_get_actions(self, text):
        return self.action_bot.get_res_json(text)
    
    def debug(self):
        # self.arm.arm_init()
        # self.set_xyz_relative(0, 100, 60, 0.5)
        while True:
            if self._stop_flag:
                return
            image = self.cap_front.read()
            res = self.crusie(image)
            det_front = self.front_det(image)
            error = res[0]
            angle = res[1]
            image = self.cap_side.read()
            det_task = self.task_det(image)
            det_hum = self.mot_hum(image)
            
            logger.info("")
            logger.info("--------------")
            logger.info("error:{} angle{}".format(error, angle))
            logger.info("front:{}".format(det_front))
            logger.info("task:{}".format(det_task))
            logger.info("hum_det:{}".format(det_hum))
            logger.info("left:{} right:{}".format(self.left_sensor.get(),self.right_sensor.get()))
            self.delay(0.5)

    def walk_lane_test(self):
        end_function = lambda: True
        self.lane_base(0.3, end_function, stop=self.STOP_PARAM)

    def close(self):
        self._stop_flag = False
        self._end_flag = True
        self.thread_key.join()
        self.cap_front.close()
        self.cap_side.close()
        # self.grap_cam.close()

    def manage(self, programs_list:list, order_index=0):

        def all_task():
            for func in programs_list:
                func()
        
        def lane_test():
            self.lane_dis_offset(0.3, 3)

        programs_suffix = [all_task, lane_test, self.task.arm.reset, self.debug]
        programs = programs_list.copy()
        programs.extend(programs_suffix)
        # print(programs)
        # 选中的python脚本序号
        # 当前选中的序号
        win_num = 5
        win_order = 0
        # 把programs的函数名转字符串
        logger.info(order_index)
        programs_str = [str(i.__name__) for i in programs]
        logger.info(programs_str)
        dis_str = sellect_program(programs_str, order_index, win_order)
        self.display.show(dis_str)

        self.stop()
        run_flag = False
        stop_flag = False
        stop_count = 0
        while True:
            # self.button_all.event()
            btn = self.key.get_btn()
            # 短按1=1,2=2,3=3,4=4
            # 长按1=5,2=6,3=7,4=8
            # logger.info(btn)
            # button_num = car.button_all.clicked()
            
            if btn != 0:
                # logger.info(btn)
                # 长按1按键，退出
                if btn == 5:
                    # run_flag = True
                    self._stop_flag = True
                    self._end_flag = True
                    break
                else:
                    if btn == 2:
                        # 序号减1
                        self.beep()
                        if order_index == 0:
                            order_index = len(programs)-1
                            win_order = win_num-1
                        else:
                            order_index -= 1
                            if win_order > 0:
                                win_order -= 1
                        # res = sllect_program(programs, num)
                        dis_str = sellect_program(programs_str, order_index, win_order)
                        self.display.show(dis_str)

                    elif btn == 4:
                        self.beep()
                        # 序号加1
                        if order_index == len(programs)-1:
                            order_index = 0
                            win_order = 0
                        else:
                            order_index += 1
                            if len(programs) < win_num:
                                win_num = len(programs)
                            if win_order != win_num-1:
                                win_order += 1
                        # res = sllect_program(programs, num)
                        dis_str = sellect_program(programs_str, order_index, win_order)
                        self.display.show(dis_str)

                    elif btn == 3:
                        # 确定执行
                        # 调用别的程序
                        dis_str = "\n{} running......\n".format(str(programs_str[order_index]))
                        self.display.show(dis_str)
                        self.beep()
                        self._stop_flag = False
                        programs[order_index]()
                        self._stop_flag = True
                        dis_str = sellect_program(programs_str, order_index, win_order)
                        self.stop()
                        self.beep()

                        # 自动跳转下一条
                        if order_index == len(programs)-1:
                            order_index = 0
                            win_order = 0
                        else:
                            order_index += 1
                            if len(programs) < win_num:
                                win_num = len(programs)
                            if win_order != win_num-1:
                                win_order += 1
                        # res = sllect_program(programs, num)
                        dis_str = sellect_program(programs_str, order_index, win_order)
                        self.display.show(dis_str)
                    logger.info(programs_str[order_index])
            else:
                self.delay(0.02)
                
            time.sleep(0.02)

        for i in range(2):
            self.beep()
            time.sleep(0.4)
        time.sleep(0.1)
        self.close()

if __name__ == "__main__":
    # kill_other_python()
    my_car = MyCar()
    my_car.lane_time(0.3, 5)
    
    # my_car.lane_dis_offset(0.3, 1.2)
    # my_car.lane_sensor(0.3, 0.5)
    # my_car.debug() 

    # text = "犯人没有带着眼镜，穿着短袖"
    # criminal_attr = my_car.hum_analysis.get_res_json(text)
    # print(criminal_attr)
    # my_car.task.reset()
    # pt_tar = my_car.task.punish_crimall(arm_set=True)
    # hum_attr = my_car.get_hum_attr(pt_tar)
    # print(hum_attr)
    # res_bool = my_car.compare_humattr(criminal_attr, hum_attr)
    # print(res_bool)
    # pt_tar = [0, 1, 'pedestrian',  0, 0.02, 0.4, 0.22, 0.82]
    # for i in range(4):
    #     my_car.set_pos_offset([0.07, 0, 0])
    #     my_car.lane_det_location(0.1, pt_tar, det="mot", side=-1)
    # my_car.close()
    # text = my_car.get_ocr()
    # print(text)
    # pt_tar = my_car.task.pick_up_ball(arm_set=True)
    # my_car.lane_det_location(0.1, pt_tar)
    
    my_car.close()
    # my_car.debug()
    # while True:
    #     text = my_car.get_ocr()
    #     print(text)

    # my_car.task.reset()
    # my_car.lane_advance(0.3, dis_offset=0.01, value_h=500, sides=-1)
    # my_car.lane_task_location(0.3, 2)
    # my_car.lane_time(0.3, 5)
    # my_car.debug()
    
    # my_car.debug()

            
    # my_car.task.pick_up_block()
    # my_car.task.put_down_self_block()
    # my_car.lane_time(0.2, 2)
    # my_car.lane_advance(0.3, dis_offset=0.01, value_h=500, sides=-1)
    # my_car.lane_task_location(0.3, 2)
    # my_car.task.pick_up_block()
    # my_car.close()
    # logger.info(time.time())
    # my_car.lane_task_location(0.3, 2)


    # my_car.debug()
    # programs = [func1, func2, func3, func4, func5, func6]
    # my_car.manage(programs)
    # import sys
    # test_ord = 0
    # if len(sys.argv) >= 2:
    #     test_ord = int(sys.argv[1])
    # logger.info("test:", test_ord)
    # car_test(test_ord)
