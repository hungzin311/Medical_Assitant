import os 

def set_proxy():
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''

def unset_proxy():
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''


