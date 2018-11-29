include getData.py

import cgi
formData = cgi.FieldStorage()

name = formData.getvalue('coin')

print "successful, please wait"

findcoin(coin)
