from collections import defaultdict
from datetime import datetime, timedelta
import locale
import json
import os

import requests

from jinja2 import Template

# Set the locale to France to get the right date format
locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')

# Load the data from the repository
r = requests.get("https://raw.githubusercontent.com/pollsposition/data/main/presidentielles/sondages.json")
polls = r.json()


# -------------------------------------------------------------------
#                       FORMAT THE NUMBERS
# -------------------------------------------------------------------

for poll in polls.values():
    for hypothese in poll['premier_tour'].values():

        if not hypothese['intentions_exprimees']:
            hypothese['intentions_exprimees'] = "?"
        else:
            hypothese['intentions_exprimees'] = int(hypothese['intentions_exprimees'] * poll['echantillon'] / 100)

        # We convert all % to strings, 0 becomes <.5%
        for candidat, intention in hypothese['intentions'].items():
            if intention < 0.5:
                formatted_intention = "-"
            else:
                formatted_intention = f"{intention}"
            hypothese['intentions'][candidat] = formatted_intention


# -------------------------------------------------------------------
#                  GROUP/ORDER BY (AND FORMAT) DATES
# 
# The date we display on the website is the date at which we add it
# to the data repository (the date that is used for modeling, however,
# is the last day the poll was on the field).
# -------------------------------------------------------------------

polls_by_date = defaultdict(list)
for sondage_id, sondage in sorted(polls.items(), key=lambda item: item[1]['date_publication'], reverse=True):
    date_raw = sondage['date_publication']
    date = datetime.strptime(date_raw, '%Y-%m-%d')
    date_display = date.strftime('%d %B %Y')

    if date.date() == datetime.today().date():
        date_display = "Ajouté aujourd'hui"

    if date.date() == datetime.today().date() - timedelta(days=1):
        date_display = "Ajouté hier"

    polls_by_date[date_display].append(sondage)


# -------------------------------------------------------------------
#                         CREATE TIMESTAMP
# -------------------------------------------------------------------

now = datetime.now()
maj = {'date': now.strftime('%d %b %Y'), 'heure': now.strftime('%H:%M')}


# -------------------------------------------------------------------
#                       FILL THE TEMPLATE
# -------------------------------------------------------------------

with open('templates/premier_tour.jinja') as file_:
    template = Template(file_.read())

if not os.path.exists('_build'):
    os.makedirs('_build')

with open('_build/premier-tour.html', 'w', encoding='utf-8') as output:
    output.write(template.render(sondage=polls_by_date, maj=maj))
