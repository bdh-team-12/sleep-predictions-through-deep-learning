#!/usr/bin/expect


spawn nsrr download [lindex $argv 0]
expect "*token*"
send "4594-J-8WjUL9-Rh3fAdUxE-B\n"
expect eof
