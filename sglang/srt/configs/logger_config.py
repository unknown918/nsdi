import logging
import colorlog

# INFO=20, DEBUG=10


A_LEVEL = 51   
B_LEVEL = 52  
C_LEVEL = 53  
D_LEVEL = 54
E_LEVEL = 55
X_LEVEL = 100
Y_LEVEL = 101 # some important message
Z_LEVEL = 105 # some important message
L_LEVEL = 136 # some important message
H_LEVEL = 137 # some important message
I_LEVEL = 138 # some important message
K_LEVEL = 139 # some important message
M_LEVEL = 140 # some important message
# 将自定义级别添加到 logging 中
logging.addLevelName(A_LEVEL, "TpModelWorker")
logging.addLevelName(B_LEVEL, "ModelRunner")
logging.addLevelName(C_LEVEL, "AttentionRunner")
logging.addLevelName(D_LEVEL, "MoeRunner")
logging.addLevelName(X_LEVEL, "Model")
logging.addLevelName(Y_LEVEL, "NOTICE")
logging.addLevelName(Z_LEVEL, "CommunicationHandler")
logging.addLevelName(L_LEVEL, "Latency")
logging.addLevelName(H_LEVEL, "ComputeLatency")
logging.addLevelName(I_LEVEL, "CommunicationLatency")
logging.addLevelName(K_LEVEL, "SchedulerLatency")
logging.addLevelName(M_LEVEL, "TimeStamp")
# 自定义日志方法
def log_a(self, message, *args, **kwargs):
    if self.isEnabledFor(A_LEVEL):
        self._log(A_LEVEL, message, args, **kwargs)

def log_b(self, message, *args, **kwargs):
    if self.isEnabledFor(B_LEVEL):
        self._log(B_LEVEL, message, args, **kwargs)

def log_x(self, message, *args, **kwargs):
    if self.isEnabledFor(X_LEVEL):
        self._log(X_LEVEL, message, args, **kwargs)

def log_y(self, message, *args, **kwargs):
    if self.isEnabledFor(Y_LEVEL):
        self._log(Y_LEVEL, message, args, **kwargs)

def log_c(self, message, *args, **kwargs):
    if self.isEnabledFor(C_LEVEL):
        self._log(C_LEVEL, message, args, **kwargs)

def log_d(self, message, *args, **kwargs):
    if self.isEnabledFor(D_LEVEL):
        self._log(D_LEVEL, message, args, **kwargs)

def log_e(self, message, *args, **kwargs):
    if self.isEnabledFor(E_LEVEL):
        self._log(E_LEVEL, message, args, **kwargs)

def log_z(self, message, *args, **kwargs):
    if self.isEnabledFor(Z_LEVEL):
        self._log(Z_LEVEL, message, args, **kwargs)

def log_l(self, message, *args, **kwargs):
    if self.isEnabledFor(L_LEVEL):
        self._log(L_LEVEL, message, args, **kwargs)

def log_h(self, message, *args, **kwargs):
    if self.isEnabledFor(H_LEVEL):
        self._log(H_LEVEL, message, args, **kwargs)

def log_i(self, message, *args, **kwargs):
    if self.isEnabledFor(I_LEVEL):
        self._log(I_LEVEL, message, args, **kwargs)

def log_k(self, message, *args, **kwargs):
    if self.isEnabledFor(K_LEVEL):
        self._log(K_LEVEL, message, args, **kwargs)

def log_m(self, message, *args, **kwargs):
    if self.isEnabledFor(M_LEVEL):
        self._log(M_LEVEL, message, args, **kwargs)

    # 将自定义日志级别方法添加到 logger 类中
logging.Logger.TpModelWorker = log_a
logging.Logger.ModelRunner = log_b
logging.Logger.AttentionRunner = log_c
logging.Logger.MoeRunner = log_d
logging.Logger.Model = log_x
logging.Logger.NOTICE = log_y
logging.Logger.MoeEngine = log_e
logging.Logger.CommunicationHandler = log_z
logging.Logger.Latency = log_l
logging.Logger.ComputeLatency = log_h
logging.Logger.CommunicationLatency = log_i
logging.Logger.SchedulerLatency = log_k
logging.Logger.TimeStamp = log_m        

def configure_logger(name: str = __name__):
    """
    配置 logger，返回一个配置好的 logger 实例，带颜色的输出。
    """
    logger = logging.getLogger(name)
    
    # 强制清除所有现有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 禁用传播到root logger，避免重复输出
    logger.propagate = False
        
    console_handler = logging.StreamHandler()
    
    # 设置彩色格式器，确保各日志级别显示不同颜色
    formatter = colorlog.ColoredFormatter(
        # '%(log_color)s%(levelname)-8s%(reset)s %(message)s [%(filename)s:%(lineno)d, %(funcName)s] ',
        '%(log_color)s%(levelname)-8s%(reset)s %(message)s',
        # '%(log_color)s%(levelname)-8s%(reset)s [%(filename)s] %(message)s',  
        log_colors={
            'DEBUG': 'blue',      
            'INFO': 'green',      
            'WARNING': 'bold_yellow',  
            'ERROR': 'red',       
            'CRITICAL': 'bold_red',
            'TpModelWorker': 'green', 
            'ModelRunner': 'blue', # cyan
            'AttentionRunner': 'blue',
            'MoeRunner': 'blue',
            'Model': 'yellow',                   
            'NOTICE': 'purple',
            'CommunicationHandler': 'cyan',
            'Latency': 'green',
            'ComputeLatency': 'yellow',
            'CommunicationLatency': 'blue',
            'SchedulerLatency': 'purple',
            'TimeStamp': 'red',
        }
    )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.setLevel(1) # 大于这个数会被输出

    return logger
