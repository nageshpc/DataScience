
import pdb
import json
import tweepy
import pickle
import sys


##############################################################
def dump_user_info_to_csv (tweets, ofh):
    ##
    print( "\t".join( ["id",   "id_str", "name", "screen_name",  "location"
            , "protected", "followers_count",  "friends_count",  "created_at"
            ,"favourites_count",  "utc_offset",  "time_zone",  "geo_enabled"
            , "verified",  "statuses_count",  "lang",  "contributors_enabled"
            , "is_translator",  "is_translation_enabled",  "profile_background_color"
            ,  "profile_background_image_url",  "profile_background_image_url_https"
            ,  "profile_background_tile",  "profile_image_url",  "profile_image_url_https"
            ,  "profile_link_color",  "profile_sidebar_border_color"
            ,  "profile_sidebar_fill_color",  "profile_text_color",  "profile_use_background_image"
            ,  "has_extended_profile",  "default_profile",  "default_profile_image",  "following"
            ,  "follow_request_sent",  "notifications",  "translator_type"
            ] ) , file=ofh)

    ## only the user info 
    ## avoid duplicates.. skip already dumped user info
    seen_users=set()
    for tweet in tweets :
        uinfo = tweet.user
        if uinfo.id not in seen_users :
            seen_users.add( uinfo.id )
        else:
            continue 
        ##
        print( uinfo.id ,  uinfo.id_str  ,uinfo.name ,uinfo.screen_name , uinfo.location 
            , uinfo.protected ,uinfo.followers_count , uinfo.friends_count , uinfo.created_at
            , uinfo.favourites_count , uinfo.utc_offset , uinfo.time_zone , uinfo.geo_enabled 
            , uinfo.verified , uinfo.statuses_count , uinfo.lang , uinfo.contributors_enabled
            , uinfo.is_translator , uinfo.is_translation_enabled , uinfo.profile_background_color
            , uinfo.profile_background_image_url , uinfo.profile_background_image_url_https 
            , uinfo.profile_background_tile , uinfo.profile_image_url , uinfo.profile_image_url_https 
            , uinfo.profile_link_color , uinfo.profile_sidebar_border_color 
            , uinfo.profile_sidebar_fill_color , uinfo.profile_text_color , uinfo.profile_use_background_image 
            , uinfo.has_extended_profile , uinfo.default_profile , uinfo.default_profile_image , uinfo.following
            , uinfo.follow_request_sent , uinfo.notifications , uinfo.translator_type  
            , file= ofh, sep ="\t" )
            #, uinfo.profile_banner_url 

    #######
##############################################################
def dump_tweet_info_to_csv (  tweets, ofh ):
    ##
    print("id created_at in_reply_to_status_id in_reply_to_user_id in_reply_to_screen_name geo coordinates place contributors is_quote_status lang pos_sen retweet_count favorite_count favorited retweeted user.id hashtags user_mentions display_urls fulltext poster_name".replace(" ", "\t")
        , file=ofh)
 
    for tweet in tweets :
        print( 
                tweet.id, tweet.created_at 
                , tweet.in_reply_to_status_id 
                , tweet.in_reply_to_user_id
                , tweet.in_reply_to_screen_name 
                , tweet.geo, tweet.coordinates , tweet.place, tweet.contributors, tweet.is_quote_status, tweet.lang
                , tweet.retweet_count, tweet.favorite_count, tweet.favorited, tweet.retweeted
                , tweet.user.id
                , ",".join( [ht["text"] for ht in tweet.entities["hashtags"] ] ) # leaving this one out for now               , ",".join( tweet.entities["symbols"]) 
                , ",".join( [um["screen_name"] for um in tweet.entities["user_mentions"]] ) 
                , ",".join( [url["display_url"] for url in tweet.entities["urls"]] ) 
               , tweet.full_text.replace("\t", "    ").replace("\n", " ")
               , tweet.user.screen_name
             , file= ofh, sep ="\t")
    ##    
#############################################################
if __name__ == "__main__":
    with open('./data/ml_tweets.bin',mode='rb') as tweet_file:
        tweets = pickle.load(tweet_file)

        #step 1... parse the tweets to get a CSV dump of twitter users
        with open("./data/tweet_users.csv", mode="w") as out_file:
            dump_user_info_to_csv(tweets , out_file) 
        #step 2... get the tweet info -- leave out the user details 
        with open("./data/tweets.csv", mode="w") as out_file:
            dump_tweet_info_to_csv(tweets, out_file)
        ##
    ###
    print("Data formatting is complete")

  
   
  
