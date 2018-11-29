<?php
include("auth.php");
if(!isset($_SESSION["username"])){
header("Location: login.php");
exit(); } else {
header("Location: logout.php");
}
?>