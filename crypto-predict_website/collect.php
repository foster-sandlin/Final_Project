<html>
<head>
</head>
<body>
<?php
 $coin = $_POST["coin"];
/* $coin = "eth";*/
/* echo $coin;*/
$output = passthru('/home/fostersandlin/anaconda3/bin/python3.6 /var/www/html/crypto-predict_website/cgi-bin/getData.py ' . $coin);
   echo $output;
    header('Location:http://35.229.59.235/crypto-predict_website/main.html');
?>
</body>
</html>
