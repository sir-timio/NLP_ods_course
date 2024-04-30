import random
import string

import numpy as np
import pandas as pd
from faker import Faker  # generates fake data
from tqdm.auto import tqdm


def get_userid(length: int = 16) -> str:
    """generate random user id

    Args:
        length (int, optional): length. Defaults to 16.

    Returns:
        str: user_id
    """
    userid = str(int(np.random.rand() * 1_000_000_000))
    for _ in range(length):
        userid = userid + random.choice(
            string.ascii_letters + str(random.randint(0, 9))
        )
    return userid


def generate_fake_social_media_url(user_name: str) -> str:
    """insert username into social media links

    Args:
        user_name (str)

    Returns:
        str: fake pii url
    """
    social_media_platforms = {
        "LinkedIn": "linkedin.com/in/",
        "YouTube": "youtube.com/c/",
        "Instagram": "instagram.com/",
        "GitHub": "github.com/",
        "Facebook": "facebook.com/",
        "Twitter": "twitter.com/",
    }
    _, domain = random.choice(list(social_media_platforms.items()))
    fake_url = f"https://{domain}{user_name}"
    return fake_url


# sometimes faker make indirect, so make it more convinient with original data
def make_simple_username(f_name: str, l_name: str, seps: list) -> str:
    """generate simple username with first and last names"""
    return (
        f"{f_name.lower()}{random.choice(seps)}{l_name.lower()}{random.randint(1,99)}"
    )


def make_simple_email(f_name: str, l_name: str, domain: str, seps: list) -> str:
    """generate simple email with first and last names"""
    return f"{f_name.lower()}{random.choice(seps)}{l_name.lower()}@{domain}"


def generate_student_info():
    """Generates all the user info (name, eamil addresses, phone number, etc) together

    Returns:
        fake student entity
    """
    # Select the student country to generate the user info based on the country
    COUNTRIES = [
        # "ru_RU",
        "en_US",
        "en_US",
        "en_US",
        "en_US",
        "en_US",
        "en_US",
        "en_US",
        "de_DE",
        "it_IT",
        "es_ES",
        "da_DK",
        "fr_FR",
        "vi_VN",
    ]
    country = random.choice(COUNTRIES)
    faker = Faker(country)
    first_name = faker.first_name()
    last_name = faker.last_name()
    user_name = faker.user_name()
    fake_url = generate_fake_social_media_url(user_name)

    seps = ["_", ".", ""]
    email = faker.email()
    if np.random.choice([0, 1]):
        email = make_simple_email(first_name, last_name, faker.domain_name(), seps)

    if np.random.choice([0, 1]):
        user_name = make_simple_username(first_name, last_name, seps)

    student = {}
    student["COUNTRY"] = country
    student["ID_NUM"] = get_userid()
    student["NAME_STUDENT"] = first_name + " " + last_name
    student["EMAIL"] = email
    student["USERNAME"] = user_name
    student["PHONE_NUM"] = faker.phone_number().replace(" ", "")
    student["URL_PERSONAL"] = fake_url
    student["STREET_ADDRESS"] = str(faker.address()).replace("\n", " ")
    del faker
    return student


if __name__ == "__main__":
    TOTAL = 10_000
    students = []
    for _ in tqdm(range(TOTAL)):
        students.append(generate_student_info())
    df = pd.DataFrame(students).reset_index(drop=True)
    df.to_csv("../../data/faker_pii.csv", index=False, encoding="UTF-8")
