این مخزن شاملکار روی مجموعه اعداد دست نویس فارسی هدی با یک شبکه ی عصبی کانولوشنی می باشد ,
که دقت زیر در دسته های مختلف بدست آمده است :
مجموعه ی Train شامل 60000 تصویر : 99.84 درصد
مجموعه ی Test شامل 20000 تصویر : 99.02 درصد
مجموعه ی Remaining شامل 22352 تصویر : 99.3 درصد
در این مخزن چهار برنامه شامل برنامه train که برای آموزش استفاده شده و 
برنامه evaluate  که برای تست مدل بعد از یادگیری روی مجموعه هدی و برنامه predict که برای تشخیص تصاویر جدید استفاده می شود .
نکته :
برنامه پیشبینی قابلیت تشخیص تصاویر ساده مانند تصاویر موجود در پوشه ی images و نه پیچیده تر را دارد .
برنامه HodaDatasetReader که برای خواندن از فایلهای دیتا ست استفاده شده است توسط آقای امیر سانیان نوشته شده و در آدرس زیر موجود است
https://github.com/amir-saniyan/HodaDatasetReader