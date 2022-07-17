



$(document).ready(function()
{
$("#storage-option").hide();
$("#introduction").attr('class','active');
var class_folder_name="folder-name";
var class_file_name="file-name";
$('#delete-button').hide();
$('#delete-button-file').hide();
var cloud_provider="";

       $(".cloud-provider").click(function(){
       $("#storage-option").show();
        $("#storage-intro").hide();
       cloud_provider=$(this).html();
      if(cloud_provider.indexOf("Microsoft") != -1){
      $('#microsoft').attr('class','active')
       $('#amazon').attr('class','')
        $('#google').attr('class','')
        $("#introduction").attr('class','');

}
else  if(cloud_provider.indexOf("Amazone") != -1){
  $('#microsoft').attr('class','');
       $('#amazon').attr('class','active');
        $('#google').attr('class','');
        $("#introduction").attr('class','');
}
else if(cloud_provider.indexOf("Google") != -1)
{
$('#microsoft').attr('class','')
       $('#amazon').attr('class','')
        $('#google').attr('class','active')
         $("#introduction").attr('class','');
}
else
{
$('#microsoft').attr('class','')
       $('#amazon').attr('class','')
        $('#google').attr('class','')
         $("#introduction").attr('class','active');
         $("#storage-option").hide();
          $("#storage-intro").show();
}

       $("#cloudProvider").val(cloud_provider);
       $('#nav-directory').html("<li><a><span>Visited directory: </span></a></li>");
         $("#current_folder_name").val("");
            $("#uploadFolder").val("");
                     $("#folder").html("");
       $("#folder").html("<div class='col-md-1 col-md-offset-5'><i style='color:#008cba;' class='fas fa-sync fa-spin fa-1x'></i></div>");


    $.ajax({
                url: '/cloud_list_directory',
                 type: 'POST',
                dataType: 'json',
                data:  JSON.stringify({
                    cloud_provider:cloud_provider
                }),
                contentType: 'application/json',

                success: function(response)
                    {

                        if(response['status']==true)
                        {
                        $('#message-file-operation').html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");

                            var is_directory=false;
                           for (var i=0;i<response['n_directory'];i++)
                           {
                           is_directory=true;
                               if(i==0)
                                   {
                                        if (response['directory_list'][i].indexOf(".")<0)
                                        {
                                        $("#folder").html('<div class="col-md-2 " ><input class="deletable" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="'+class_folder_name+'">'+response['directory_list'][i]+'</a></div>');
                                        }
                                        else
                                        {
                                        $("#folder").html('<div class="col-md-2 " ><input class="deletable-file" id="selected-folder-'+response['directory_list'][i]+'"   value="'+response['directory_list'][i]+'"  type="checkbox"  /><i style="color:#008cba;padding:20px;" class="fas fa-file fa-pull-top fa-3x"></i><br /><a class="'+class_file_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }
                                   }
                               else
                                   {
                                   if (response['directory_list'][i].indexOf(".")<0)
                                        {
                                        $("#folder").append('<div class="col-md-2 " ><input class="deletable" id="selected-folder-'+response['directory_list'][i]+'"   value="'+response['directory_list'][i]+'"  type="checkbox"  /><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="'+class_folder_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }
                                        else
                                        {
                                        $("#folder").append('<div class="col-md-2 " ><input class="deletable-file" id="selected-folder-'+response['directory_list'][i]+'"   value="'+response['directory_list'][i]+'"  type="checkbox"  /><i style="color:#008cba;padding:20px;" class="fas fa-file fa-pull-top fa-3x"></i><br /><a class="'+class_file_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }

                                    }

                            }
                            if(!is_directory)
                            {
                             $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                      $("#folder").html("");
                            }
                        }
                        else
                            {



                                $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
    $("#folder").html("");

                            }
                    },
                error: function(error)
                    {
                                $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                                $("#folder").html("");
                                $("#current_folder_name").val("");
                                 $("#uploadFolder").val("");

                    }
            });

});

var state_delete=".folder-name";
$('#delete-file-folder').click(function(){

    if (state_delete==".folder-name")
    {
    $('#delete-button').show();
   $('.folder-name').toggleClass('folder-name delete-allow-file-folder');
   class_folder_name='delete-allow-file-folder';
   state_delete=".delete-allow-file-folder";
   $('#enable-delete-option').toggleClass('fa fa-toggle-off fa fa-toggle-on');


   }
   else
   {

    $('#delete-button').hide();
   $('.delete-allow-file-folder').toggleClass('delete-allow-file-folder folder-name ');
   $('#enable-delete-option').toggleClass('fa fa-toggle-on fa fa-toggle-off');
   state_delete=".folder-name";
   class_folder_name='folder-name';
   }

});



var state_delete_file='.file-name';
$('#delete-file-folder-file').click(function(){

    if (state_delete_file==".file-name")
    {
    $('#delete-button-file').show();
   $('.file-name').toggleClass('file-name delete-allow-file-folder-file');
   class_file_name='delete-allow-file-folder-file';
   state_delete_file=".delete-allow-file-folder-file";
   $('#enable-delete-option-file').toggleClass('fa fa-toggle-off fa fa-toggle-on');


   }
   else
   {

    $('#delete-button-file').hide();
   $('.delete-allow-file-folder-file').toggleClass('delete-allow-file-folder-file file-name');
   $('#enable-delete-option-file').toggleClass('fa fa-toggle-on fa fa-toggle-off');
   state_delete_file=".file-name";
   class_file_name='file-name';
   }

});


$(document).delegate('.delete-allow-file-folder','click', function(){
var id="#selected-folder-"+$(this).html();

var is_check=$(id).prop('checked');

$(id).prop('checked', !is_check);
});


$(document).delegate('.delete-allow-file-folder-file','click', function(){
var id="selected-folder-"+$(this).html();

var is_check=document.getElementById(id).checked;
//alert(is_check)
document.getElementById(id).checked=!is_check;
});

$("#delete-folder-btn-confirm").click(function(){
$('#delete-folder-status').html("");
var user_input=$('#user-confirmation-folder').val();
if (user_input!="confirm")
{
$('#delete-folder-status').html("<div class='alert alert-info'>You must write confirm in text box to proceed</div>");

return false;
}
elements_check_box=document.getElementsByClassName('deletable');
var folder_names=""
for( var i=0;i<elements_check_box.length;i++)
{
if (elements_check_box.item(i).checked)
{
folder_names=folder_names+elements_check_box.item(i).value+";";
}
}
if(folder_names.length<=0)
{
$('#delete-folder-status').html("<div class='alert alert-info'>Select a folder</div>");

return false;
}
$('#delete-folder-status').html("<div class='alert alert-success' role='alert'>Request accepted deletion started <a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>")

var directory=$("#current_folder_name").val();
 $.ajax({
                url: '/delete_folder',
                 type: 'POST',
                dataType: 'json',
                data:  JSON.stringify({
                    cloud_provider:cloud_provider,
                    folder_names:folder_names,
                    directory:directory,

                }),
                contentType: 'application/json',
                success: function(response)
                    {

                        if(response['status']==true)
                        {
                         $('#delete-folder-status').html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");

                        $('#message-file-operation').html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                      $("#folder").html("");
                       var is_directory=false;
                           for (var i=0;i<response['n_directory'];i++)
                           {
                           is_directory=true;
                               if(i==0)
                                   {
                                        if (response['directory_list'][i].indexOf(".")<0)
                                        {
                                        $("#folder").html('<div class="col-md-2 " ><input class="deletable" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="'+class_folder_name+'">'+response['directory_list'][i]+'</a></div>');
                                        }
                                        else
                                        {
                                        $("#folder").html('<div class="col-md-2 " ><input class="deletable-file" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:#008cba;padding:20px;" class="fas fa-file fa-pull-top fa-3x"></i><br /><a class="'+class_file_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }
                                   }
                               else
                                   {
                                   if (response['directory_list'][i].indexOf(".")<0)
                                        {
                                        $("#folder").append('<div class="col-md-2 " ><input class="deletable" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="'+class_folder_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }
                                        else
                                        {
                                        $("#folder").append('<div class="col-md-2 " ><input class="deletable-file" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:#008cba;padding:20px;" class="fas fa-file fa-pull-top fa-3x"></i><br /><a class="'+class_file_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }

                                    }

                            }
                            if(!is_directory)
                            {

$('#delete-folder-status').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");


                             $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                                 $("#folder").html("");
                            }
                        }
                        else
                            {


$('#delete-folder-status').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");



                                $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                                //$("#folder").html("");


                            }
                    },
                error: function(error)
                    {
            $("#folder").html("");
                    }
            });
});



$("#delete-file-btn-confirm").click(function(){
$('#delete-file-status').html("");
var user_input=$('#user-confirmation-file').val();
if (user_input!="confirm")
{
$('#delete-file-status').html("<div class='alert alert-info'>You must write confirm in text box to proceed</div>");

return false;
}
elements_check_box=document.getElementsByClassName('deletable-file');
var file_names=""
for( var i=0;i<elements_check_box.length;i++)
{
if (elements_check_box.item(i).checked)
{
file_names=file_names+elements_check_box.item(i).value+";";
}
}
if(file_names.length<=0)
{
$('#delete-folder-status').html("<div class='alert alert-info'>Select a file</div>");

return false;
}
$('#delete-file-status').html("<div class='alert alert-success' role='alert'>Request accepted deletion started <a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>")
 $.ajax({
                url: '/delete_file',
                 type: 'POST',
                dataType: 'json',
                data:  JSON.stringify({
                    cloud_provider:cloud_provider,
                    file_names:file_names,
                     directory:$("#current_folder_name").val(),
                }),
                contentType: 'application/json',

                success: function(response)
                    {

                        if(response['status']==true)
                        {
                        $('#message-file-operation').html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                        $('#delete-file-status').html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");

                      $("#folder").html("");

                      var is_directory=false;
                           for (var i=0;i<response['n_directory'];i++)
                           {
                           is_directory=true;
                               if(i==0)
                                   {
                                        if (response['directory_list'][i].indexOf(".")<0)
                                        {
                                        $("#folder").html('<div class="col-md-2 " ><input class="deletable" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="'+class_folder_name+'">'+response['directory_list'][i]+'</a></div>');
                                        }
                                        else
                                        {
                                        $("#folder").html('<div class="col-md-2 " ><input class="deletable-file" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:#008cba;padding:20px;" class="fas fa-file fa-pull-top fa-3x"></i><br /><a class="'+class_file_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }
                                   }
                               else
                                   {
                                   if (response['directory_list'][i].indexOf(".")<0)
                                        {
                                        $("#folder").append('<div class="col-md-2 " ><input class="deletable" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="'+class_folder_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }
                                        else
                                        {
                                        $("#folder").append('<div class="col-md-2 " ><input class="deletable-file" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:#008cba;padding:20px;" class="fas fa-file fa-pull-top fa-3x"></i><br /><a class="'+class_file_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }

                                    }

                            }
                             if(!is_directory)
                            {
                            $('#delete-file-status').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");

                             $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                                 $("#folder").html("");
                            }
                        }
                        else
                            {

$('#delete-file-status').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");


                                $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                                //$("#folder").html("");


                            }
                    },
                error: function(error)
                    {
            $("#folder").html("");
                    }
            });

});



 $(document).delegate('.folder-name', 'click', function()  //enabling accpeting folder name to list directory
 {

  var clicked_folder_name=$(this).html();

  var current_folder_name_from_input_box=$("#current_folder_name").val();

  $("#current_folder_name").val(current_folder_name_from_input_box+clicked_folder_name+"/");
 var folder_name=current_folder_name_from_input_box+clicked_folder_name;
 $("#uploadFolder").val(folder_name);
  //alert(folder_name);
  var visited_path=folder_name.split('/');

    access_path=""
  for(var i=0;i<visited_path.length;i++)
  {
  access_path=access_path+visited_path[i]+"/";
  //alert(access_path);
  //alert(visited_path[i]);
  if(i==0)
  {
   $('#nav-directory').html("<li><a><span>Visited directory: </span></a></li><li><a><span id="+access_path+" class='back-directory'>"+visited_path[i]+"</span></a></li>");
}
else
{
   $('#nav-directory').append("<li><a><span id="+access_path+" class='back-directory'>"+visited_path[i]+"</span></a></li>");
}
  }



       $("#folder").html("<div class='col-md-1 col-md-offset-5'><i style='color:#008cba;' class='fas fa-sync fa-spin fa-1x'></i></div>");


    $.ajax({
                url: '/list_directory',
                 type: 'POST',
                dataType: 'json',
                data:  JSON.stringify({
                    cloud_provider:cloud_provider,
                    folder_name:folder_name,
                }),
                contentType: 'application/json',

                success: function(response)
                    {

                        if(response['status']==true)
                        {
                        $('#message-file-operation').html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                      $("#folder").html("");
                      var is_directory=false;
                           for (var i=0;i<response['n_directory'];i++)
                           {
                           is_directory=true;
                               if(i==0)
                                   {
                                        if (response['directory_list'][i].indexOf(".")<0)
                                        {
                                        $("#folder").html('<div class="col-md-2 " ><input class="deletable" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="'+class_folder_name+'">'+response['directory_list'][i]+'</a></div>');
                                        }
                                        else
                                        {
                                        $("#folder").html('<div class="col-md-2 " ><input class="deletable-file" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:#008cba;padding:20px;" class="fas fa-file fa-pull-top fa-3x"></i><br /><a class="'+class_file_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }
                                   }
                               else
                                   {
                                   if (response['directory_list'][i].indexOf(".")<0)
                                        {
                                        $("#folder").append('<div class="col-md-2 " ><input class="deletable" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="'+class_folder_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }
                                        else
                                        {
                                        $("#folder").append('<div class="col-md-2 " ><input class="deletable-file" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:#008cba;padding:20px;" class="fas fa-file fa-pull-top fa-3x"></i><br /><a class="'+class_file_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }

                                    }

                            }
                            if(!is_directory)
                            {

                                $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                                $("#folder").html("");

                            }
                        }
                        else
                            {



                                $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                                $("#folder").html("");


                            }
                    },
                error: function(error)
                    {
            $("#folder").html("");
                    }
            });



 });

$("#save-folder").click(function()
{
var upload_folder_name=$("#current_folder_name").val();
var folder_name=$("#new-folder-name").val();
$("#new-folder-status").html("<div class='col-md-1 col-md-offset-5'><i style='color:#008cba;' class='fas fa-sync fa-spin fa-1x'></i></div>");


$.ajax({
                url: '/create_folder',
                 type: 'POST',
                dataType: 'json',
                data:  JSON.stringify({
                    cloud_provider:cloud_provider,
                    upload_folder_name:upload_folder_name,
                    folder_name:folder_name,
                }),
                contentType: 'application/json',

                success: function(response)
                    {

                        if(response['status']==true)
                        {
                        $("#new-folder-name").val("");
                            $('#message-file-operation').html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                             $("#new-folder-status").html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>")

                           $("#folder").append('<div class="col-md-2 " ><input class="deletable" id="selected-folder-'+response['folder_name']+'"  value="'+response['folder_name']+'" type="checkbox"  /><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="'+class_folder_name+'">'+response['folder_name']+'</a></div>');




                        }




                        else
                            {



                            $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                              $("#new-folder-status").html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>")


                            }
                    },
                error: function(error)
                    {

                    }
            });
});

$('#refresh-folder').click(function(){
 var folder_name=$("#current_folder_name").val();
 $("#folder").html("<div class='col-md-1 col-md-offset-5'><i style='color:#008cba;' class='fas fa-sync fa-spin fa-1x'></i></div>");

  $.ajax({
                url: '/list_directory',
                 type: 'POST',
                dataType: 'json',
                data:  JSON.stringify({
                    cloud_provider:cloud_provider,
                    folder_name:folder_name,
                }),
                contentType: 'application/json',

                success: function(response)
                    {

                        if(response['status']==true)
                        {
                            $('#message-file-operation').html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                            $("#folder").html("");
                            var is_directory=false;
                           for (var i=0;i<response['n_directory'];i++)
                           {
                           is_directory=true;
                               if(i==0)
                                   {
                                        if (response['directory_list'][i].indexOf(".")<0)
                                        {
                                        $("#folder").html('<div class="col-md-2 " ><input class="deletable" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="'+class_folder_name+'">'+response['directory_list'][i]+'</a></div>');
                                        }
                                        else
                                        {
                                        $("#folder").html('<div class="col-md-2 " ><input class="deletable-file" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:#008cba;padding:20px;" class="fas fa-file fa-pull-top fa-3x"></i><br /><a class="'+class_file_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }
                                   }
                               else
                                   {
                                   if (response['directory_list'][i].indexOf(".")<0)
                                        {
                                        $("#folder").append('<div class="col-md-2 " ><input class="deletable" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="'+class_folder_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }
                                        else
                                        {
                                        $("#folder").append('<div class="col-md-2 " ><input class="deletable-file"  id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:#008cba;padding:20px;" class="fas fa-file fa-pull-top fa-3x"></i><br /><a class="'+class_file_name+'">'+response['directory_list'][i]+'</a></div>');



                                        }

                                    }

                            }
                            if (!is_directory)
                            {
                               $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                               $("#folder").html("");
                            }

                        }
                        else
                            {



                            $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                               $("#folder").html("");

                            }
                    },
                error: function(error)
                    {

                    }
            });
});
 $(document).delegate('.back-directory', 'click', function(){
 var current_folder_name=$(this).attr('id');
 $("#current_folder_name").val(current_folder_name);
 $("#uploadFolder").val(current_folder_name);

  $("#folder").html("<div class='col-md-1 col-md-offset-5'><i style='color:#008cba;' class='fas fa-sync fa-spin fa-1x'></i></div>");
    var folder_name=current_folder_name;


    $.ajax({
                url: '/list_directory',
                 type: 'POST',
                dataType: 'json',
                data:  JSON.stringify({
                    cloud_provider:cloud_provider,
                    folder_name:folder_name,
                }),
                contentType: 'application/json',

                success: function(response)
                    {

                        if(response['status']==true)
                        {
                            $("#folder").html("");
                            $('#message-file-operation').html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                            var is_directory=false;
                           for (var i=0;i<response['n_directory'];i++)
                           {
                           is_directory=true;
                               if(i==0)
                                   {
                                        if (response['directory_list'][i].indexOf(".")<0)
                                        {
                                        $("#folder").html('<div class="col-md-2 " ><input class="deletable" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="'+class_folder_name+'">'+response['directory_list'][i]+'</a></div>');
                                        }
                                        else
                                        {
                                        $("#folder").html('<div class="col-md-2 " ><input class="deletable-file" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  /><i style="color:#008cba;padding:20px;" class="fas fa-file fa-pull-top fa-3x"></i><br /><a class="'+class_file_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }
                                   }
                               else
                                   {
                                   if (response['directory_list'][i].indexOf(".")<0)
                                        {
                                        $("#folder").append('<div class="col-md-2 " ><input class="deletable" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox" /><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="'+class_folder_name+'">'+response['directory_list'][i]+'</a></div>');

                                        }
                                        else
                                        {
                                        $("#folder").append('<div class="col-md-2 " ><input class="deletable-file" id="selected-folder-'+response['directory_list'][i]+'"  value="'+response['directory_list'][i]+'" type="checkbox"  ><i style="color:#008cba;padding:20px;" class="fas fa-file fa-pull-top fa-3x"></i><br /><a class="'+class_file_name+'">'+response['directory_list'][i]+'</a></div>');



                                        }

                                    }

                            }
                            if(!is_directory)
{
$('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                               $("#folder").html("");
}
                        }
                        else
                            {



                            $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                               $("#folder").html("");

                            }
                    },
                error: function(error)
                    {

                    }
            });

 });
/*
$("#upload-files-server").click(function(){
alert("button is working");
alert($("#files-to-upload").val());
 var upload_folder_name=$("#current_folder_name").val();
 alert(upload_folder_name+cloud_provider);
 $.ajax({
                url: '/upload_files',
                 type: 'POST',
                data:   new FormData($('#files-to-upload')[0]),

                success: function(response)
                    {

                        if(response['status']==true)
                        {

                           for (var i=0;i<response['n_directory'];i++)
                           {
                               if(i==0)
                                   {
                                        $("#folder").html('<div class="col-md-2 " ><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="folder-name">'+response['directory_list'][i]+'</a></div>');
                                   }
                               else
                                   {
                                        $("#folder").append('<div class="col-md-2 " ><i style="color:rgb(222, 193, 29);padding:20px;" class="fas fa-folder fa-pull-top fa-3x"></i><br /><a  class="folder-name">'+response['directory_list'][i]+'</a></div>');
                                    }

                            }
                        }
                        else
                            {
                            $("#folder").html(response['message']);
                            }
                    },
                error: function(error)
                    {

                    }
            });

});*/

$('#selected-file').change(function(){
 $('#file-upload-status').html("");

})

 $('#upload-file-btn').click(function() {
 var upload_folder_name=$("#current_folder_name").val();
        var form_data = new FormData($('#upload-file')[0]);
       if ($('#selected-file')[0].files.length === 0)
        {

         $('#file-upload-status').html("<div class='alert alert-info' role='alert'> please select a file..<a class='close'   onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");

        }
        else
        {

       // alert(upload_folder_name);
        //$("#uploadFolder").val(upload_folder_name);
 $("#file-upload-status").html("<div class='col-md-1 col-md-offset-5'><i style='color:#008cba;' class='fas fa-sync fa-spin fa-1x'></i></div>");



        $.ajax({
            type: 'POST',
            url: '/upload_files',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(response) {
            $("#file-upload-status").html("");
            if (response['status']==true)
            {
            $('#selected-file').val("");
            $('#message-file-operation').html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");

 $('#file-upload-status').html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");
                    for (var i=0;i<response['n_file_uploaded'];i++)
                           {


                                        $("#folder").append('<div class="col-md-2 " ><input class="deletable-file" id="selected-folder-'+response['uploaded_files_on_cloud']+'"  value="'+response['uploaded_files_on_cloud']+'" type="checkbox"  /><i style="color:#008cba;padding:20px;" class="fas fa-file fa-pull-top fa-3x"></i><br /><a class="'+class_file_name+'">'+response['uploaded_files_on_cloud'][i]+'</a></div>');





                            }

               }
               else
               {

              $('#message-file-operation').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");

            $('#file-upload-status').html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>");

               }
            }
        });
        }
    });

 });

