from selenium import webdriver
from selenium.webdriver.common.by import By


def test_lambdatest_simple_form_demo():
    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.get("https://www.lambdatest.com/selenium-playground/simple-form-demo")
    driver.find_element(By.XPATH, "//input[@id='user-message']").send_keys("Pytest is a Test Framework")
    driver.find_element(By.XPATH, "//button[text()='Get Checked Value']").click()
    message = driver.find_element(By.ID, "message").text
    assert message == "Pytest is a Test Framework"
