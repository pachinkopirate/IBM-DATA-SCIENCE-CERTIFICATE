'''
Assignment: SQL Notebook for Peer Assignment
SQL queries
!pip install sqlalchemy==1.3.9
!pip install ibm_db_sa
!pip install ipython-sql
'''

#Let us first load the SQL extension and establish a connection with the database
%load_ext sql

'''
Task 1
Display the names of the unique launch sites in the space mission
'''
%%sql
SELECT DISTINCT(launch_site)
FROM SPACEXTBL;

'''
Task 2
Display 5 records where launch sites begin with the string 'CCA'
'''
%%sql
SELECT *
FROM SPACEXTBL
WHERE launch_site LIKE 'CCA%''
limit 5
;


# Task 3
# Display the total payload mass carried by boosters launched by NASA (CRS)

%%sql

SELECT sum(payload_mass__kg_)

FROM SPACEXTBL
WHERE customer = 'NASA (CRS)'


;

'''
Task 4
Display average payload mass carried by booster version F9 v1.1
'''

%%sql

SELECT AVG(payload_mass__kg_)

FROM SPACEXTBL
WHERE booster_version LIKE 'F9 v1.1%'


;

'''
Task 5
List the date when the first successful landing outcome in ground pad was acheived.
Hint:Use min function
'''
%%sql

SELECT MIN(DATE)
FROM SPACEXTBL
WHERE landing__outcome LIKE '%Success%'

'''
Task 6
List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000
'''

%%sql

SELECT booster_version
FROM SPACEXTBL
WHERE 4000 < payload_mass__kg_
and payload_mass__kg_ < 6000
and landing__outcome LIKE '%Success%drone ship%'
;

'''
Task 7
List the total number of successful and failure mission outcomes
'''
%%sql

SELECT COUNT(*)
FROM SPACEXTBL
WHERE landing__outcome LIKE '%Success%'
;
SELECT COUNT(*)
FROM SPACEXTBL
WHERE landing__outcome NOT LIKE '%Success%'
;

'''
Task 8
List the names of the booster_versions which have carried the maximum payload mass. Use a subquery
'''
%%sql

SELECT booster_version
FROM SPACEXTBL
WHERE payload_mass__kg_ =
(
SELECT max(payload_mass__kg_) as max_pl
FROM SPACEXTBL
    )
;
'''
Task 9
List the failed landing_outcomes in drone ship, their booster versions, and launch site names for in year 2015
'''
%%sql
SELECT booster_version, launch_site, landing__outcome
FROM SPACEXTBL
WHERE landing__outcome LIKE '%Failure%drone%'
and DATE LIKE '%2015%'
;


# Task 10
# Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order

%%sql

SELECT landing__outcome, count(*)
FROM SPACEXTBL
WHERE DATE BETWEEN '2010-06-04' and '2017-03-20'
GROUP BY landing__outcome
ORDER BY 2 DESC

;
