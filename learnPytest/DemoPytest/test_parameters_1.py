import math

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By


@pytest.mark.parametrize("num1, num2, expected_total", [("25", "25", "50"), ("10", "10", "30"), ("30", "40", "70")])
def test_lambdatest_two_input_fields(num1, num2, expected_total):
    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.get("https://www.lambdatest.com/selenium-playground/simple-form-demo")
    driver.find_element(By.XPATH, "//input[@placeholder='Please enter first value']").send_keys(num1)
    driver.find_element(By.XPATH, "//input[@placeholder='Please enter second value']").send_keys(num2)
    driver.find_element(By.XPATH, "//button[contains(text(),'Get Sum')]").click()
    actual_result = driver.find_element(By.ID, "addmessage").text
    assert actual_result == expected_total, "Actual & Expected Totals Do Not Match"


@pytest.mark.parametrize("base", [1, 2, 3])
@pytest.mark.parametrize("exponent", [4, 5, 6])
def test_raising_base_to_power(base, exponent):
    result = base ** exponent
    assert result == math.pow(base, exponent)