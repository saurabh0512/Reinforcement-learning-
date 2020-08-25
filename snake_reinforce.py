

from tkinter import *
import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU

size=300
snake_width=30
colour='sienna4'
epsilon = 0.1

class mov_snake():
    def __init__(self):
        self.window = Tk()
        self.window.title("Snake")
        self.canvas = Canvas(self.window , bg='alice blue', width = size , height= size)
        self.snake = self.canvas.create_rectangle(120, 120 ,150, 150, fill='Black')
        #MOUSE COORDINATES
        self.coordinates= []
        #SNAKE BODY+HEAD COORDINATES
        self.my_variables=[self.canvas.coords(self.snake)]
        self.mouse =self.canvas.create_rectangle(self.mouse_coordinates() , fill =colour)
        self.canvas.pack()

        self.x = 0
        self.y = 0
        self.count = 0
        self.my_rectangles=[]
        self.mode = 'in'
        self.replay= False
        self.reward = 0
        
        self.curr_value = 1
        self.value= 1
        self.not_allowed = {3: 1, 1 : 3, 0:2, 2:0}
        self.reverse = 0
        
        self.initialize_grid()
        #self.window.play=Button(self.window, text='Play', callback=self.play_game())
        #self.window.play.pack()
        
    def mainloop(self):
        self.window.mainloop()
            
    
    def reset_game(self):
        self.canvas = Canvas(self.window , bg='alice blue', width = size , height= size)
        self.snake = self.canvas.create_rectangle(120, 120, 150, 150, fill='Black') 
        self.coordinates= []
        self.my_variables=[self.canvas.coords(self.snake)]
        self.mouse =self.canvas.create_rectangle(self.mouse_coordinates() , fill =colour)
        self.canvas.pack()
        self.x = 0
        self.y = 0
        self.count = 0
        self.my_rectangles=[]
        self.mode = 'in'
        self.reverse = 0
        
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
                if self.value == 3:
                    self.x=-(snake_width)
                    self.y= 0
                    self.curr_value = 3
            
                elif self.value == 1:
                    self.x = (snake_width)
                    self.y = 0
                    self.curr_value = 1
            
                elif self.value == 2:
                    self.x=0
                    self.y=(snake_width)
                    self.curr_value = 2
                    
                elif self.value == 0:
                    self.x = 0
                    self.y = -(snake_width)
                    self.curr_value = 0
            else :
                self.reverse = 1
                
            if ((self.canvas.coords(self.snake)[0]< size and self.canvas.coords(self.snake)[2] > 0) and (self.canvas.coords(self.snake)[1] < size and self.canvas.coords(self.snake)[3] > 0)) :
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
               # self.canvas.after(200, self.initialize_game)
            else :
                self.mode = 'out'
                self.replay = False
                self.end_game()
        
    
    def update_snake(self):
        if  ( self.canvas.coords(self.snake) == self.coordinates):
            self.mode = 'eat'
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
    
    #UP:0 , RIGHT:1, DOWN:2 , LEFT:3
    def key_press(self, move):
        keys=[0,1,2,3]
        
        if (move in keys):
            self.value = move
    
    def get_reward(self):
        if self.mode == 'out':
            self.reward = -0.8
        elif self.mode == 'in':
            if self.reverse == 1:
                self.reward = -0.6
            else:
                self.reward = -0.2
        elif self.mode == 'intercept':
            self.reward = -0.7
        elif self.mode == 'eat':
            self.reward = 10
        return self.reward
    
    def get_bound(self, direction):
        if (self.canvas.coords(self.snake)[direction] +((-1)**(direction)) *(snake_width) == 0 ):
            return False
        if self.canvas.coords(self.snake) in self.my_variables :
            return False
        return True
    
    def get_mouse(self):
        dist = np.array(self.coordinates) - np.array(self.canvas.coords(self.snake))
        food_direction = np.zeros(4, dtype=int)
        if dist[0]>0:
            food_direction[1] = 1
        if dist[0]<0:
            food_direction[0] = 1
        if dist[1]>0:
            food_direction[3] = 1
        if dist[1]<0:
            food_direction[2] = 1
        return food_direction
    
    def get_env(self):
        self.state = np.zeros(8, dtype=int)
        direction = {0:1, 1:2, 2:3, 3:0}
        for i in range(4):
            self.state[i] = not self.get_bound(direction[i])
        self.state[4:] = self.get_mouse()
        self.state = self.state.reshape((1,-1))
        return self.state
            
#-----------------------------        
          #MODEL
#-----------------------------
          
def build_model(lr=0.001):
    model = Sequential()
    model.add(Dense(8, input_shape=(8,)))
    model.add(PReLU())
    model.add(Dense(8))
    model.add(PReLU())
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse')
    return model

#-----------------------------
            #Experience
#-----------------------------
            
class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, env):
        return self.model.predict(env)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, 8))
        targets = np.zeros((data_size, 4))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            env, action, reward, env_next, game_over = self.memory[j]
            inputs[i] = env
           
            targets[i] = self.predict(env)
        
            Q_sa = np.max(self.predict(env_next))
            
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets
        

#--------------------------
        #Training
#--------------------------
        
        
def train(my_model, data_size):
    n_epochs = 500
    
    game = mov_snake()
    model = my_model
    experience = Experience(model)
    
    moves=[]
    
    for i in range(n_epochs):
        loss = 0
        
        n_episodes = 0
        
        episode_mov=[]
        
        if game.replay == False:
            game.replay = True
            game.reset_game()
        
        env = game.get_env()
        
        action = 1
        
        while game.replay == True:
            valid_actions = {0:[0,1,3] ,1:[0,1,2], 2:[1,2,3], 3:[0,2,3]}
            
            prev_env = env
            
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions[action])
            else:
                action = np.argmax(experience.predict(prev_env))
            
            game.value = action
            game.initialize_game()
            
            episode_mov.append(action) 
            if (game.mode == 'out') or (game.mode == 'intercept'):
                game.replay == False
                break
            else :
                game.replay == True
            
            reward = game.get_reward()
            env = game.get_env()
            
            episode = [prev_env, action, reward, env, game.replay]
            experience.remember(episode)
            n_episodes += 1
            
            score = game.count
            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0)
            loss = model.evaluate(inputs, targets, verbose=0)
        
        moves.append(episode_mov)
        
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Score: {:d} "
        print(template.format(i, n_epochs-1, loss, n_episodes, score))
        
my_model = build_model()

train(my_model, 10)
                  
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    