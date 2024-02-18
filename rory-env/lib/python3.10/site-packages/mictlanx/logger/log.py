from collections.abc import Mapping
import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
import json
import threading
from typing import Any
from option import Some,NONE,Option
# from lo
# from pathlib import Path

class DumbLogger(object):
    def debug(self,**kargs):
        return
    def info(self,**kargs):
        return
    def error(self,**kargs):
        return


class JsonFormatter(logging.Formatter):
    def format(self, record):
        thread_id = threading.current_thread().getName()
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger_name': record.name,
            "thread_name":thread_id
        }
        if isinstance(record.msg, dict):
            log_data.update(record.msg)  # Add the dictionary data to the log
        else:
            log_data['message'] = record.getMessage()

        return json.dumps(log_data,indent=4,)+"\n"




class Log(logging.Logger):
    def __init__(self,
                 formatter:logging.Formatter=JsonFormatter(),
                 name:str ="mictlanx-client-0",
                 level:int = logging.DEBUG,
                 path:str = "/mictlanx/client",
                 disabled:bool = False,
                 console_handler_filter =lambda record: record.levelno == logging.DEBUG,
                 file_handler_filter =lambda record: record.levelno == logging.INFO,
                 console_handler_level:int = logging.DEBUG,
                 file_handler_level:int = logging.INFO,
                #  format:str = '%(asctime)s %(levelname)s %(threadName)s %(message)s',
                #  extension:str = "log",
                 error_log:bool = False,
                 filename:Option[str] = NONE,
                 output_path:Option[str] =NONE,
                 error_output_path:Option[str] = NONE,
                 create_folder:bool= True,
                 to_file:bool = True,
                 when:str = "m",
                 interval:int = 10
                #  "/mictlanx/client/mictlanx-client-0.error", 
                 ):
        super().__init__(name,level)
        if (not os.path.exists(path) and create_folder) :
            os.makedirs(path)
            
        if not (disabled):
            consolehanlder =logging.StreamHandler(sys.stdout)
            consolehanlder.setFormatter(formatter)
            consolehanlder.setLevel(console_handler_level)
            consolehanlder.addFilter(console_handler_filter)
            self.addHandler(consolehanlder)
            if to_file:
                # filehandler = logging.FileHandler(filename= output_path.unwrap_or("{}/{}.log".format(path,filename.unwrap_or(name))))
                filehandler = TimedRotatingFileHandler(
                    filename= output_path.unwrap_or("{}/{}".format(path,filename.unwrap_or(name))),
                    when=when,
                    interval=interval
                    # backupCount=
                )
                filehandler.setFormatter(formatter)
                filehandler.setLevel(file_handler_level)
                filehandler.addFilter(file_handler_filter)
                self.addHandler(filehandler)
            # 
            if error_log:
                errorFilehandler = logging.FileHandler(filename=error_output_path.unwrap_or("{}/{}.error".format(path,filename.unwrap_or(name))))
                errorFilehandler.setFormatter(formatter)
                errorFilehandler.setLevel(logging.ERROR)
                errorFilehandler.addFilter(lambda record: record.levelno == logging.ERROR)
                self.addHandler(errorFilehandler)
    
    # def add_handler(self,handler:logging.Handler):
        # self.addHandler()