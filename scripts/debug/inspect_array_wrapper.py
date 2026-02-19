import vectorbt as vbt

try:
    print(f"ArrayWrapper settings: {vbt.settings.array_wrapper}")
except AttributeError:
    print("vbt.settings.array_wrapper not found")
