$("#go").click(function() {
    $ytUrl = $('#input').val();
    if ($ytUrl == "")
        return;
    $.post("/video", {'youtube_url': $ytUrl}, function(response) {
        var myDiv = $('#video'); // The place where you want to inser the template
        myDiv.html(response);
    });
});
