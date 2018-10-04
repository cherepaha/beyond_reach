import pandas as pd
import Tkinter as tkr
from datetime import datetime

class WalkingExpInfoUI(tkr.Frame):    
    def __init__(self, master=None):
        tkr.Frame.__init__(self, master)
        self.remaining_ids_file_name = 'remaining_subj_ids.csv'
        self.pack()
        self.createWidgets()
        
    def createWidgets(self):   
        self.subj_id_label = tkr.Label(self, text='Participant number').pack()        
        self.subj_id_entry = tkr.Entry(self)
        self.subj_id_entry.pack()

        self.order_label = tkr.Label(self, text='Task order').pack()
        self.order_var = tkr.StringVar(value='mw')
        self.mouse_first_radio = tkr.Radiobutton(self, text="Computer first", padx=20, 
                                                 variable=self.order_var, value='mw')
        self.mouse_first_radio.pack(anchor=tkr.W)
        self.walking_first_radio = tkr.Radiobutton(self, text="Walking first", padx=20, 
                                                   variable=self.order_var, value='wm')
        self.walking_first_radio.pack(anchor=tkr.W)
        
        self.task_label = tkr.Label(self, text='Current task').pack()
        self.task_var = tkr.StringVar(value='mouse')
        self.mouse_radio = tkr.Radiobutton(self, text="Computer", padx=20, 
                                           variable=self.task_var, value='mouse')
        self.mouse_radio.pack(anchor=tkr.W)
        self.walking_radio = tkr.Radiobutton(self, text="Walking", padx=20, 
                                             variable=self.task_var, value='walking')
        self.walking_radio.pack(anchor=tkr.W)
        
        self.generate_button = tkr.Button(self, text='Generate', command=self.generate)
        self.generate_button.pack() 
        
        self.start_button = tkr.Button(self, text='Start experiment', command=self.proceed)
        self.start_button.pack()       

    def generate(self):
        self.subj_id_entry.delete(0,tkr.END)

        self.remaining_ids = pd.read_csv(self.remaining_ids_file_name)
        self.current_id = self.remaining_ids.sample(1)
        
        self.subj_id_entry.insert(0,int(self.current_id['subj_id']))
        order = self.current_id['order'].values[0]
        self.order_var.set(order)
        self.task_var.set('mouse' if order=='mw' else 'walking')
        
        self.start_button.bind('<Button-1>', self.drop_id)

    def drop_id(self, event):
        if int(self.current_id['subj_id']) == int(self.subj_id_entry.get()):
            self.remaining_ids = self.remaining_ids.drop(self.current_id.index)
            self.remaining_ids.to_csv(self.remaining_ids_file_name, index=False)
            print(self.current_id)
            print('ID dropped')

    def proceed(self):
        self.exp_info = {'subj_id': int(self.subj_id_entry.get()),
                         'order': self.order_var.get(),
                         'task': self.task_var.get(),
                         'start_time': datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')}
        self.quit()
            