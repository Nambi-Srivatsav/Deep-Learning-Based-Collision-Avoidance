import pygame
from pygame.color import THECOLORS
import pdb
import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import from_pygame, to_pygame
from pymunk.pygame_util import DrawOptions as draw
import pymunk.util as u
import random
import math
import numpy as np
from make_it_learn import *

width = 600
height = 600
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
wall_direction = 1
summary_sensor_data = []

step_size_value = 1/10
clock_tick_value = 1000
bot_speed = 50

model = None

next_action = 0


def points_from_angle(angle):
    """ Returns the unit vector with given angle """
    return math.cos(angle),math.sin(angle)
    
class Bot_env:
        
    def __init__(self):
        self.crashed = False
        self.detect_crash = 0
        self.space = pymunk.Space()
        self.build_bot(50, 450, 20)
        self.num_steps = 0
        self.walls = []
        self.wall_shapes = []
        self.wall_rects = []
        
        wall_body, wall_shape, wall_rect = self.build_wall(200, 50, 50)
        self.wall_rects.append(wall_rect)
        wall_body, wall_shape, wall_rect = self.build_wall(200, 125, 50)
        self.wall_rects.append(wall_rect)
        wall_body, wall_shape, wall_rect = self.build_wall(200, 550, 50)
        self.wall_rects.append(wall_rect)
        wall_body, wall_shape, wall_rect = self.build_wall(200, 450, 50)
        self.wall_rects.append(wall_rect)
        wall_body, wall_shape, wall_rect = self.build_wall(400, 350, 50)
        self.wall_rects.append(wall_rect)
        wall_body, wall_shape, wall_rect = self.build_wall(400, 250, 50)
        self.wall_rects.append(wall_rect)
        wall_body, wall_shape, wall_rect = self.build_wall(500, 250, 50)
        self.wall_rects.append(wall_rect)
        wall_body, wall_shape, wall_rect = self.build_wall(600, 250, 50)
        self.wall_rects.append(wall_rect)
        wall_body, wall_shape, wall_rect = self.build_wall(115, 950, 400)
        self.wall_rects.append(wall_rect)
        
        self.prev_action = 0
        
    
    def build_wall(self, x, y, r):
        """ Builds the wall """
        size = r
        wall_rect = pygame.Rect(x-r,600-y-r, 2*r, 2*r)
        return wall_rect,wall_rect,wall_rect
        

    def build_bot(self, x, y, r):
        """ Builds the bot """
        size = r
        box_points = list(map(Vec2d, [(-size, -size), (-size, size), (size,size), (size, -size)]))
        mass  = 0.5
        moment = pymunk.moment_for_poly(mass,box_points, Vec2d(0,0))
        self.bot = pymunk.Body(mass, moment)
        self.bot.position = Vec2d(x,y)
        self.bot.angle = 1.54
        bot_direction = Vec2d(points_from_angle(self.bot.angle))
        self.space.add(self.bot)
        self.bot_rect = pygame.Rect(x-r,600-y-r, 2*r, 2*r)

        return self.bot
    
    def draw_everything(self,flag=0):
        
        img = pygame.image.load("./assets/intel.jpg")
        x, y = 580,550
        adjusted_img_position = (x-50,y+50)
        screen.blit(img,to_pygame(adjusted_img_position,screen))
        
        if(flag==0 and self.detect_crash == 0):
            (self.bot_rect.x,self.bot_rect.y) = self.bot.position[0],600-self.bot.position[1]
            self.circle_rect = pygame.draw.circle(screen, (169,169,169), (self.bot_rect.x,self.bot_rect.y), 20, 0)
        
        elif(flag==0 and self.detect_crash >= 1):
            
            (self.bot_rect.x,self.bot_rect.y) = self.bot.position[0],600-self.bot.position[1]
            self.circle_rect = pygame.draw.circle(screen, (0,255,0), (self.bot_rect.x,self.bot_rect.y), 20, 0)
        
        else:
            (self.bot_rect.x,self.bot_rect.y) = self.bot.position[0],600-self.bot.position[1]
            self.circle_rect = pygame.draw.circle(screen, (255,0,0), (self.bot_rect.x,self.bot_rect.y), 20, 0)
        
        img = pygame.image.load("./assets/spherelight.png")
        offset = Vec2d(img.get_size()) / 2.
        x, y =  self.bot.position
        y = 600.0 -y
        
        adjusted_img_position = (x,y) - offset
        screen.blit(img,adjusted_img_position)
        
        for ob in self.wall_rects:
            pygame.draw.rect(screen, (169,169, 169), ob)
    
    def plan_angle(self,A,B):
        """ Angle between two points """
        angle = np.arctan2(B[1] - A[1], B[0] - A[0])
        return angle
        
    
    def _step(self, action, crash_step=0):
        """ Take the simulation one step further """
            
        self.bot.angle = self.bot.angle % 6.2831853072
        
        if action == 3:  
            self.bot.angle -= 0.1
            self.prev_body_angle =  self.bot.angle
            self.bot_direction = Vec2d(points_from_angle(self.bot.angle))
            bot_direction = self.bot_direction
            if(crash_step > 0):
                self.bot.velocity = bot_speed/3 * bot_direction
            else:
                self.bot.velocity = bot_speed * bot_direction
                
            self.prev_action = 3
            
        elif action == 4:
            self.bot.angle += 0.1
            self.prev_body_angle =  self.bot.angle
                
            self.bot_direction = Vec2d(points_from_angle(self.bot.angle))
            bot_direction = self.bot_direction
            self.bot.velocity = bot_speed * bot_direction
            if(crash_step == 1):
                self.bot.velocity = bot_speed/3 * bot_direction
            else:
                self.bot.velocity = bot_speed * bot_direction
            
            self.prev_action = 4
                 
        elif action == 5:
            
            self.bot.angle += 0.
            self.prev_body_angle =  self.bot.angle
                
            self.bot_direction = Vec2d(points_from_angle(self.bot.angle))
            bot_direction = self.bot_direction
            self.bot.velocity = bot_speed * bot_direction
            if(crash_step == 1):
                self.bot.velocity = bot_speed/3 * bot_direction
            else:
                self.bot.velocity = bot_speed * bot_direction
        
        screen.fill(THECOLORS["white"])
        
        self.draw_everything()
        self.space.step(step_size_value)
        
        clock.tick(clock_tick_value)
        
        x, y = self.bot.position
        sensors_data = self.all_sensor_sensors_data(x, y, self.bot.angle)
        normalized_sensors_data = [(x-100.0)/100.0 for x in sensors_data] 
        state = np.array([normalized_sensors_data])
        
        sensors_data = np.append(sensors_data,math.degrees(self.bot.angle))
        sensors_data = np.append(sensors_data,[0])
       
        print(sensors_data[:-2])
        
        data_tensor = torch.Tensor(sensors_data[:-1]).view(1,-1)

        for ob in self.wall_rects:
            if ob.colliderect(self.circle_rect):
                    self.recover_from_crash(bot_direction)
        
        if (x >= 580 or x <= 20 or y <= 20 or y >=680):
                    self.recover_from_crash(bot_direction)
        
        signal_data = sensors_data[:-2]
        
        if 1 in signal_data:
        
            if(action == 5):
                action = self.prev_action
            self.crashed = True
            sensors_data[-1] = 1
            summary_sensor_data.append(sensors_data)
            print(sensors_data[:-2])
            reward = -500
            self.recover_from_crash(bot_direction)
        else:
            
            self.detect_crash = 0
            summary_sensor_data.append(sensors_data)
            
        
            
        return


    def recover_from_crash(self, bot_direction):
        """ What happens when bot crashes """
        while self.crashed:
            self.crashed = False 
            for i in range(1):

                self.bot.angle += 3.14
                self.bot_direction = Vec2d(points_from_angle(self.bot.angle))
                bot_direction = self.bot_direction
                self.bot.velocity = bot_speed * bot_direction
            
                screen.fill(THECOLORS["white"])
                self.draw_everything(flag=1)
                self.space.step(step_size_value)
                pygame.display.flip()
                clock.tick(clock_tick_value)
                    

    def all_sensor_sensors_data(self, x, y, angle):
        """ Returns the all sensor values """
        
        sensors_data = []
       
        middle_sensor_start_point = (25 + x, y) # sensor start point
        middle_sensor_end_point = (65 + x , y) # sensor end point
        number_of_sensors = 5
        relative_angles = []
        angle_to_begin_with = 1.3
        offset_increment =  (angle_to_begin_with*2)/(number_of_sensors-1) 
        relative_angles.append(-angle_to_begin_with) # angle to begin with
        
        for i in range(number_of_sensors-1):
            relative_angles.append(relative_angles[i]+offset_increment)
        
        sensor_list = []
        
        for i in range(number_of_sensors):
            sensor_list.append([middle_sensor_start_point,middle_sensor_end_point, relative_angles[i]])
            sensors_data.append(self.sensor_reading(sensor_list[i], x, y, angle))
       
        pygame.display.update()
        return sensors_data

    def sensor_reading(self, sensor, x, y, angle):
        """ Returns the reading for a single sensor """
        
        distance = 0
        (x1,y1) = sensor[0][0],sensor[0][1]
        (x2,y2) = sensor[1][0],sensor[1][1]
        sensor_angle = sensor[2]
        pixels_in_path = []
        number_of_points = 10
        
        for k in range(number_of_points):
            x_new = x1 + (x2-x1) * (k/number_of_points)
            y_new = y1 + (y2-y1) * (k/number_of_points)
            pixels_in_path.append((x_new,y_new))

        for pixel in pixels_in_path:
            distance += 1
            
            pixel_in_game = self.rotate((x, y), (pixel[0], pixel[1]), angle + sensor_angle)
                
            sensor_start_in_game = self.rotate((x, y), (x1, pixels_in_path[-1][1]), angle + sensor_angle)
            sensor_end_in_game = self.rotate((x, y),  pixels_in_path[-1], angle + sensor_angle)
            
            if pixel_in_game[0] <= 0 or pixel_in_game[1] <= 0 or pixel_in_game[0] >= width or pixel_in_game[1] >= height:
                return distance
                
            else:
                for ob in self.wall_rects:
                    if ob.collidepoint((pixel_in_game[0],pixel_in_game[1])):
                        return distance

        
        # Draw the sensor
        pygame.draw.line(screen,(30,144,255),sensor_start_in_game,sensor_end_in_game)
        return distance


        
    def rotate(self,origin, point, angle):
        """ Rotates a point along a given point """
         
        x1, y1 = origin
        x2, y2 = point
        final_x = x1 + math.cos(angle) * (x2 - x1) - math.sin(angle) * (y2 - y1)
        final_y = y1 + math.sin(angle) * (x2 - x1) + math.cos(angle) * (y2 - y1)
        final_y = abs(width - final_y)
        
        return final_x,final_y
        

def take_left_or_right_turn(env):
    """ Take random action for left or right turn """
    x = random.randint(3,4)
    for i in range(14):
        env._step(x)


def go_straight(env):
    """ Take action to go straight """

    x = random.randint(5,5)
    for i in range(14):
        env._step(x)
        

if __name__ == "__main__":
    
    env = Bot_env()
    random.seed(10)
    
    env._step(3)
    
    for i in range(3000):
        
        if(env.bot.position[0] > 500 and env.bot.position[1] > 520):
            print("MISSION COMPLETE!")
            exit()
            
        else:
            if (random.random() > 0.5):
                take_left_or_right_turn(env)
            else:
                go_straight(env)
        
        np.savetxt("./sensor_data/sensor_data.txt",summary_sensor_data)

