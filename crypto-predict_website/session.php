<?php
session_start();
if(!isset($_SESSION["username"])){
header("Location: http://3merch.com\login.php");
exit();}
?>