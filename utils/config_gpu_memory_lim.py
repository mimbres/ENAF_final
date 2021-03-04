import tensorflow as tf

def allow_gpu_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            #logical_devices = tf.config.list_logical_devices('GPU') 
    except:
        # Invalid device or cannot modify logical devices once initialized.
        pass
    return


# Deprecated!!
# def config_gpu_memory_limit(size_Gb):
#     mem_size = round(1024 * size_Gb)
#     gpus = tf.config.list_physical_devices('GPU')
    
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_virtual_device_configuration(
#                     gpu,
#                     [tf.config.experimental.VirtualDeviceConfiguration(
#                         memory_limit=mem_size)])
#         except RuntimeError as e:
#             print(e)


                
def config_gpu_memory_limit(size_Gb):
    mem_size = round(1024 * size_Gb)   
    physical_devices = tf.config.list_physical_devices('GPU') 
    try:
        for device in physical_devices:
            tf.config.set_logical_device_configuration(device,
                [tf.config.LogicalDeviceConfiguration(memory_limit=mem_size)])
        #logical_devices = tf.config.list_logical_devices('GPU') 
    except:
        # Invalid device or cannot modify logical devices once initialized.
        pass 
    
    
"""    
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.set_logical_device_configuration(physical_devices[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=9000)])
"""
