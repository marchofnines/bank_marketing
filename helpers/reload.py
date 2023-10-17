import sys
sys.path.append('/Users/basilhaddad/jupyter/module17/bank_marketing_repo/')
from importlib import reload

#from mypackage.module1 import my_function
# Now you can just use:
#my_function()
#or use import mypackge.module1 as mm
#mm.my_function()

"""
When we need to update my reload:
1. Check if it is sys.modules

import sys
print('helpers.reload' in sys.modules)  # Replace with the actual module name

2. or just restart kernel
3. ...?
"""

"""def myreload():
    import helpers.preprocessing as pp 
    import helpers.plots as myplt
    import helpers.tools as tools
    reload(pp)
    reload(myplt)
    reload(tools)
    print("Reloaded helpers.preprocessing and helpers.plots.")"""
    
    
def myreload():
    global pp, myplt, tools  # Declare as global variables
    import helpers.preprocessing as pp 
    import helpers.plot as plot
    import helpers.tools as tools
    reload(pp)
    reload(plot)
    reload(tools)
    print("Reloaded helpers.preprocessing, helpers.plots, and helpers.tools.")
