import os 

def set_proxy():
    # os.environ['HTTP_PROXY'] = 'http://10.61.11.42:3128'
    # os.environ['HTTPS_PROXY'] = 'http://10.61.11.42:3128'
    pass

def unset_proxy():
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''

