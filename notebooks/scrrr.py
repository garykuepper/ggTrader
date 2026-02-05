import geckodriver_autoinstaller
from selenium import webdriver

geckodriver_autoinstaller.install() # Automatically installs the compatible geckodriver
driver = webdriver.Firefox()