# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:41:18 2020

@author: Saurabh
"""
from tkinter import *
import random

size=300
snake_width=30
colour='sienna4'

class mov_snake():
    def __init__(self):
        self.window = Tk()
        self.window.title("Snake")
        self.canvas = Canvas(self.window , bg='alice blue', width = size , height= size)
        self.snake = self.canvas.create_rectangle(120, 120 ,150, 150, fill='Black')
        self.coordinates= []

        self.my_variables=[self.canvas.coords(self.snake)]
        self.mouse =self.canvas.create_rectangle(self.mouse_coordinates() , fill =colour)
        self.canvas.pack()

        self.x = 0
        self.y = 0
        self.count = 0
        self.my_rectangles=[]
        self.mode = 'in'
        self.replay= False
        
        
        self.curr_value = 'Right'
        self.value= 'Right'
        self.not_allowed = {'Left' : 'Right', 'Right' : 'Left', 'Up':'Down', 'Down':'Up'}
        
        self.initialize_grid()
        
        self.window.play=Button(self.window, text='Play', callback=self.play_game())
        self.window.play.pack()
        
    def mainloop(self):
        self.window.mainloop()
            
    
    def reset_game(self):
        self.canvas = Canvas(self.window , bg='alice blue', width = size , height= size)
        self.snake = self.canvas.create_rectangle(120, 120, 150, 150, fill='Black') 
        self.coordinates= []
        self.my_variables=[self.canvas.coords(self.snake)]
        self.mouse =self.canvas.create_rectangle(self.mouse_coordinates() , fill =colour)
        self.canvas.pack()
        self.initialize_grid()
        
    def play_game(self):
        self.replay = True
        self.window.bind('<Key>' , self.key_press)
        self.initialize_game()
    # ---------------------------------------    
    # Initialization Functions
    # ---------------------------------------
        
    def initialize_grid(self):
        for i in range(0, 9):
            self.canvas.create_line((i+1)*(snake_width), 0, (i+1)*(snake_width), size)
        
        for i in range(0, 9):
            self.canvas.create_line(0, (i+1)*(snake_width), size, (i+1)*(snake_width))
        
    def mouse_coordinates(self):
        list1 = [(i*(snake_width), j*(snake_width)) for i in range(0, int(size/snake_width)) for j in range(0,int(size/snake_width))]
        for i in self.my_variables:
            if (i[0],i[1]) in list1:
                list1.remove((i[0],i[1]))
        x,y = random.choice(list1)
        self.coordinates.extend((x, y, x+30, y+30))
        return self.coordinates
        
    
    # ---------------------------------------
    # Game Play Function
    #----------------------------------------
    #get apple coordinates if encountered change coordinates of snake

    def initialize_game(self):
        if self.replay :
            if self.not_allowed[self.curr_value] != self.value:
                if self.value == 'Left':
                    self.x=-(snake_width)
                    self.y= 0
                    self.curr_value = 'Left'
            
                elif self.value == 'Right':
                    self.x = (snake_width)
                    self.y = 0
                    self.curr_value = 'Right'
            
                elif self.value == 'Down':
                    self.x=0
                    self.y=(snake_width)
                    self.curr_value = 'Down'
                    
                elif self.value == 'Up':
                    self.x = 0
                    self.y = -(snake_width)
                    self.curr_value = 'Up'
                
            if ((self.canvas.coords(self.snake)[0]< size and self.canvas.coords(self.snake)[2] > 0) and 
                (self.canvas.coords(self.snake)[1] < size and self.canvas.coords(self.snake)[3] > 0)) :
                self.mode = 'in'
                a, b, c, d = self.canvas.coords(self.snake)
                self.canvas.move(self.snake, self.x, self.y)
                self.update_snake()
                
                if self.count > 0:
                    if self.canvas.coords(self.snake) in self.my_variables:
                        self.mode = 'intercept'
                        self.replay = False
                        self.end_game()
                    else :
                        self.my_variables.clear()
                        self.my_variables.append([a, b, c ,d])
                        for i in range(self.count):
                            self.move_rect(i)
                self.canvas.after(200, self.initialize_game)
            else :
                self.mode = 'out'
                self.replay = False
                self.end_game()
        
    
    def update_snake(self):
        if  ( self.canvas.coords(self.snake) == self.coordinates):
            self.coordinates.clear()
            self.count += 1
            self.my_rectangles.append(self.canvas.create_rectangle(0, 0, 0, 0,fill='Black'))
            self.canvas.coords(self.mouse, self.mouse_coordinates())
            
    def move_rect(self, i):
        self.my_variables.append(self.canvas.coords(self.my_rectangles[i]))
        self.canvas.coords(self.my_rectangles[i], self.my_variables[i])
        
    def end_game(self):
        self.canvas.delete('all')
        self.canvas.create_text(size/2, size/4, font="cmr 30 bold", text="GAME OVER")

        self.canvas.create_text(size/2, size/2, font="cmr 30 bold", text='SCORE :' + str(self.count)+'\n')
        
    def key_press(self,event):
        keys=['Up','Down','Right','Left']
        
        if (event.keysym in keys):
            self.value = event.keysym
    
game = mov_snake()
game.mainloop()