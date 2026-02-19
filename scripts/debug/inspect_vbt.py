import vectorbt as vbt

print(f"Type of vbt.settings.caching: {type(vbt.settings.caching)}")
print(f"Dir of vbt.settings.caching: {dir(vbt.settings.caching)}")

try:
    print(vbt.settings.caching.fields)
except AttributeError as e:
    print(f"Error accessing fields: {e}")

# Check for context manager capabilities
try:
    with vbt.settings.caching(enabled=False):
        print("Context manager caching(...) works.")
except Exception as e:
    print(f"Context manager caching(...) failed: {e}")
